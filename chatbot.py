import uuid
import os
import io
import time
from functools import lru_cache
from dotenv import load_dotenv
from database import SessionLocal, ChatMessage
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Distance, VectorParams,
    Filter, FieldCondition, MatchValue, PointIdsList
)
from sentence_transformers import SentenceTransformer
from groq import Groq
import pdfplumber
from tabulate import tabulate
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import warnings
warnings.filterwarnings("ignore", message="Could get FontBBox from font descriptor*")

# Configure Tesseract path (Windows specific)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()

# Initialize clients
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "chatbot_sessions"
PDF_COLLECTION_NAME = "pdf_documents"
MAX_HISTORY_LENGTH = 5
SUMMARY_CACHE_SIZE = 100
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize DePlot
device = "cuda" if torch.cuda.is_available() else "cpu"
deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)

def create_collections():
    """Initialize Qdrant collections if they don't exist"""
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    if COLLECTION_NAME not in existing_collections:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            timeout=1200
        )
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="session_id",
            field_schema="keyword"
        )
    
    if PDF_COLLECTION_NAME not in existing_collections:
        qdrant.recreate_collection(
            collection_name=PDF_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            timeout=1200
        )
        qdrant.create_payload_index(
            collection_name=PDF_COLLECTION_NAME,
            field_name="document_id",
            field_schema="keyword"
        )

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def store_message(session_id, role, message):
    """Store message in both database and vector store"""
    db = SessionLocal()
    chat_record = ChatMessage(session_id=session_id, role=role, message=message)
    db.add(chat_record)
    db.commit()
    db.refresh(chat_record)
    db.close()

    # Store in vector database
    embedding = embedder.encode(message).tolist()
    point = PointStruct(
        id=int(uuid.uuid4().int % 1e12),
        vector=embedding,
        payload={
            "session_id": session_id,
            "role": role,
            "message": message,
            "timestamp": int(time.time())
        }
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

    # Clean up old messages
    existing = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id))
        ]),
        limit=100,
        with_payload=True
    )
    if len(existing[0]) > MAX_HISTORY_LENGTH:
        old_points = sorted(existing[0], key=lambda x: x.payload.get("timestamp", 0))
        old_ids = [p.id for p in old_points[:-MAX_HISTORY_LENGTH]]
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=old_ids)
        )

@lru_cache(maxsize=SUMMARY_CACHE_SIZE)
def get_conversation_summary(session_id):
    """Generate a concise summary of the conversation"""
    db = SessionLocal()
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.id).all()
    db.close()

    if not messages:
        return "No previous conversation history"

    conversation = "\n".join(
        f"{msg.role}: {msg.message}" for msg in messages[-10:]
    )

    summary_prompt = (
        "Create a very concise summary (1-2 sentences max) focusing on:\n"
        "1. Main topic being discussed\n"
        "2. Any specific numbers/dates mentioned\n"
        "3. The most recent question\n\n"
        "Conversation:\n" + conversation
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return "Current conversation context unavailable"

def get_session_history(session_id):
    """Retrieve conversation history from vector store"""
    result = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id))
        ]),
        limit=MAX_HISTORY_LENGTH,
        with_payload=True
    )
    messages = sorted(result[0], key=lambda x: x.payload.get("timestamp", 0))
    return [{"role": p.payload["role"], "content": p.payload["message"]} for p in messages]

def extract_pdf_content(pdf_path):
    """Extract text and images from PDF"""
    full_text = ""
    images = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"

            tables = page.extract_tables()
            for table in tables:
                formatted_table = tabulate(table, headers="firstrow", tablefmt="grid")
                full_text += f"\n\nTABLE:\n{formatted_table}\n\n"

            if page.images:
                page_image = page.to_image(resolution=300)
                for img in page.images:
                    try:
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        cropped = page_image.original.crop(bbox)
                        images.append(cropped)
                    except Exception as e:
                        print(f"Image extraction failed: {e}")
    
    return full_text, images

def extract_chart_data(image: Image.Image) -> str:
    """Extract text from chart images using OCR"""
    try:
        image = image.convert("L")
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        chart_text = pytesseract.image_to_string(image, config="--psm 6")
        
        if chart_text.strip():
            return f"Chart contains: {chart_text.strip()}"
        else:
            width, height = image.size
            return f"Visual chart approximately {width}x{height} pixels with data points"
    except Exception as e:
        return f"[Chart content could not be extracted: {str(e)}]"

def extract_charts_with_deplot(pdf_path: str, document_id: str, chunk_size: int = 500):
    """
    Extract charts from PDF using DePlot and store in vector database
    
    Args:
        pdf_path: Path to PDF file
        document_id: Unique document identifier
        chunk_size: Size for text chunks
    
    Returns:
        List of processing results
    """
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract and process image
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Extract table data
                text_table = "Extract all data from this chart in table format with clear headers."
                inputs_table = deplot_processor(images=image, text=text_table, return_tensors="pt").to(device)
                table_ids = deplot_model.generate(**inputs_table, max_new_tokens=512)
                table_data = deplot_processor.decode(table_ids[0], skip_special_tokens=True)
                
                # Generate summary
                text_summary = ("Provide a comprehensive summary of this chart including: "
                              "1. Chart title and type, 2. Key trends and patterns, "
                              "3. Notable data points, 4. Overall conclusion.")
                inputs_summary = deplot_processor(images=image, text=text_summary, return_tensors="pt").to(device)
                summary_ids = deplot_model.generate(**inputs_summary, max_new_tokens=512)
                chart_summary = deplot_processor.decode(summary_ids[0], skip_special_tokens=True)
                
                # Create and store chunks
                combined_content = f"CHART SUMMARY:\n{chart_summary}\n\nEXTRACTED DATA:\n{table_data}"
                chunks = []
                current_chunk = ""
                
                for para in [p for p in combined_content.split('\n') if p.strip()]:
                    if len(current_chunk) + len(para) + 1 <= chunk_size:
                        current_chunk += para + "\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n"
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Store chunks in vector database
                points = []
                for i, chunk in enumerate(chunks):
                    embedding = embedder.encode(chunk).tolist()
                    point = PointStruct(
                        id=int(uuid.uuid4().int % 1e12),
                        vector=embedding,
                        payload={
                            "document_id": document_id,
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "type": "chart_chunk",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "content": chunk,
                            "full_summary": chart_summary,
                            "full_table": table_data
                        }
                    )
                    points.append(point)
                
                if points:
                    qdrant.upsert(collection_name=PDF_COLLECTION_NAME, points=points)
                
                results.append({
                    "page": page_num + 1,
                    "image_index": img_index + 1,
                    "summary": chart_summary,
                    "table_data": table_data,
                    "num_chunks": len(chunks)
                })
                
            except Exception as e:
                print(f"❌ Error processing page {page_num+1} image {img_index+1}: {str(e)}")
                continue
    
    return results

def store_pdf_chunks(text: str, document_id: str):
    """Store PDF text content in vector database"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < 1000:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    for chunk in chunks:
        embedding = embedder.encode(chunk).tolist()
        point = PointStruct(
            id=int(uuid.uuid4().int % 1e12),
            vector=embedding,
            payload={
                "document_id": document_id,
                "content": chunk,
                "source": "pdf"
            }
        )
        qdrant.upsert(collection_name=PDF_COLLECTION_NAME, points=[point])

def process_pdf(pdf_path: str):
    """Process a PDF file and store its contents"""
    text, images = extract_pdf_content(pdf_path)
    ocr_text = ""
    chart_summaries = []

    for i, image in enumerate(images):
        try:
            ocr_text += pytesseract.image_to_string(image)
            chart_summary = extract_chart_data(image)
            chart_summaries.append(f"Chart {i+1}: {chart_summary}")
        except Exception as e:
            print(f"Image processing failed: {e}")
            chart_summaries.append(f"Chart {i+1}: [Content not extracted]")

    full_text = (
        "PDF TEXT CONTENT:\n" + text + 
        "\n\nIMAGE TEXT CONTENT:\n" + ocr_text + 
        "\n\nCHART SUMMARIES:\n" + "\n".join(chart_summaries)
    )
    
    document_id = os.path.basename(pdf_path)
    store_pdf_chunks(full_text, document_id)

    # Process charts with DePlot
    deplot_results = extract_charts_with_deplot(pdf_path, document_id)
    print(f"✅ DePlot processed {len(deplot_results)} charts")

def get_relevant_context(user_message: str, session_id: str):
    """Retrieve relevant context from vector stores"""
    question_embedding = embedder.encode(user_message).tolist()
    
    # Search PDF content
    pdf_results = qdrant.search(
        collection_name=PDF_COLLECTION_NAME,
        query_vector=question_embedding,
        limit=10,
        score_threshold=0.4
    )
    
    # Get conversation history
    history = get_session_history(session_id)
    recent_history = history[-3:]
    
    pdf_context = "\n".join([hit.payload.get("content", "") for hit in pdf_results])
    history_context = "\n".join([msg["content"] for msg in recent_history])
    
    return pdf_context, history_context

def get_verified_context(session_id):
    """Retrieve messages containing numerical data"""
    db = SessionLocal()
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.id.desc()).limit(10).all()
    db.close()
    
    return [msg for msg in messages if any(char.isdigit() for char in msg.message)]

def chat_with_session(session_id, user_message):
    """Main chat function with context-aware responses"""
    try:
        uuid_obj = uuid.UUID(session_id)
    except ValueError:
        return "❌ Invalid session ID format. Please generate a valid session."

    # Get all context sources
    conversation_summary = get_conversation_summary(session_id)
    pdf_context, history_context = get_relevant_context(user_message, session_id)
    verified_contexts = get_verified_context(session_id)
    verified_text = "\n".join([msg.message for msg in verified_contexts])

    # Construct system prompt
    system_prompt = (
        "You are a context-aware assistant. Follow these rules strictly:\n"
        "1. CONVERSATION SUMMARY:\n" + conversation_summary + "\n\n"
        "2. Maintain context for follow-up questions\n"
        "3. DOCUMENT CONTEXT:\n" + (pdf_context if pdf_context else "None") + "\n\n"
        "4. VERIFIED NUMERICAL CONTEXT:\n" + (verified_text if verified_text else "None") + "\n\n"
        "5. Respond clearly and concisely to the latest user query while maintaining continuity.\n"
    )

    # Prepare messages for LLM
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_session_history(session_id)[-3:])
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return "Sorry, I couldn't generate a response at this time."

    # Store conversation
    store_message(session_id, "user", user_message)
    store_message(session_id, "assistant", reply)
    get_conversation_summary.cache_clear()

    return reply

# Initialize collections on startup
create_collections()