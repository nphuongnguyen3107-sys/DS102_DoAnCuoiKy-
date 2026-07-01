# ml_pipeline/rag_engine.py
import os
import re
import glob
from pypdf import PdfReader

# Các từ dừng tiếng Việt phổ biến để loại bỏ khỏi từ khóa tìm kiếm
VIETNAMESE_STOPWORDS = {
    "là", "thì", "mà", "ở", "có", "và", "của", "cho", "được", "bị", "các", "những", 
    "một", "với", "trong", "theo", "đến", "về", "ra", "này", "kia", "đó", "nào", "gì"
}

class RAGEngine:
    def __init__(self, workspace_dir=None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.chunks = []  # Danh sách lưu các đoạn văn bản: [{"text": ..., "page": ..., "source": ...}]
        self.pdf_loaded = False
        self.loaded_files = []
        
        # Tự động quét và tải file PDF khi khởi tạo
        self.auto_load_pdfs()

    def auto_load_pdfs(self):
        pdf_files = []
        
        # 1. Quét trong thư mục gốc
        pdf_pattern = os.path.join(self.workspace_dir, "*.pdf")
        pdf_files.extend(glob.glob(pdf_pattern))
        
        # 2. Quét trong thư mục docs/
        docs_pattern = os.path.join(self.workspace_dir, "docs", "*.pdf")
        pdf_files.extend(glob.glob(docs_pattern))
        
        # 3. Quét thêm cả thư mục cha nếu app.py chạy ở thư mục con
        if not pdf_files:
            pdf_pattern_parent = os.path.join(os.path.dirname(self.workspace_dir), "*.pdf")
            pdf_files.extend(glob.glob(pdf_pattern_parent))
            
            docs_pattern_parent = os.path.join(os.path.dirname(self.workspace_dir), "docs", "*.pdf")
            pdf_files.extend(glob.glob(docs_pattern_parent))

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            # Bỏ qua các file báo cáo đồ án của sinh viên (thường bắt đầu bằng [report])
            if filename.lower().startswith("[report]") or "2452" in filename:
                print(f"[RAG] Skipping project report: {filename}")
                continue
                
            print(f"[RAG] Reading guideline file: {filename}...")
            self.load_pdf(pdf_path)

    def load_pdf(self, pdf_path):
        filename = os.path.basename(pdf_path)
        if not os.path.exists(pdf_path):
            print(f"[RAG] File does not exist: {filename}")
            return
            
        try:
            reader = PdfReader(pdf_path)
            
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text()
                if not text:
                    continue
                
                # Phân tách trang thành các đoạn văn dựa trên ký tự xuống dòng kép hoặc ngắt dòng hợp lý
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                
                for p in paragraphs:
                    # Lưu thông tin chunk
                    self.chunks.append({
                        "text": p,
                        "page": page_num,
                        "source": filename
                    })
                    
            self.pdf_loaded = True
            self.loaded_files.append(filename)
            print(f"[RAG] Loaded {filename} ({len(reader.pages)} pages, {len(self.chunks)} chunks).")
        except Exception as e:
            print(f"[RAG] Error reading PDF {filename}: {e}")

    def clean_text(self, text):
        """Làm sạch văn bản, chuyển về chữ thường và xóa ký tự đặc biệt."""
        text = text.lower()
        # Thay thế các ký tự đặc biệt thành khoảng trắng
        text = re.sub(r'[^\w\s\dàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        return text

    def search(self, query, top_k=4):
        if not self.pdf_loaded or not self.chunks:
            return []
            
        # Lazily fit TF-IDF vectorizer if not already fitted
        if not hasattr(self, 'vectorizer'):
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Convert stopwords list to match sklearn expectations
            self.vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
            corpus = [self.clean_text(chunk["text"]) for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        from sklearn.metrics.pairwise import cosine_similarity
        
        # Vectorize query and calculate cosine similarity
        query_clean = self.clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        scored_chunks = []
        for idx, similarity in enumerate(similarities):
            if similarity > 0:
                chunk = self.chunks[idx]
                # Apply small boost for key medical terms in the chunk if they are also in the query
                boost = 0.0
                chunk_text_clean = self.clean_text(chunk["text"])
                for term in [
                    "kháng_sinh", "kháng", "vi_khuẩn", "phác_đồ", "điều_trị",
                    "nhạy", "đề_kháng", "cephalosporin", "fluoroquinolone", 
                    "carbapenem", "penicillin", "aminoglycoside", "tetracycline",
                    "ciprofloxacin", "levofloxacin", "imipenem", "meropenem",
                    "gentamicin", "amikacin", "streptomycin", "colistin", "mcr",
                    "gyra", "parc", "ctx-m", "tem", "oxa", "shv", "ndm", "kpc"
                ]:
                    # Clean comparison of word bounds
                    term_clean = term.replace("_", " ")
                    if term_clean in query_clean and term_clean in chunk_text_clean:
                        boost += 0.05
                
                scored_chunks.append((similarity + boost, chunk))
                
        # Sắp xếp các đoạn theo điểm số giảm dần
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top_k kết quả tốt nhất
        results = []
        seen_texts = set()
        
        for score, chunk in scored_chunks:
            text_summary = chunk["text"][:100]
            if text_summary not in seen_texts:
                results.append(chunk)
                seen_texts.add(text_summary)
                if len(results) >= top_k:
                    break
                    
        return results
