import os
import re
from pathlib import Path
from docx import Document
from pypdf import PdfReader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss 

# --- Configuration (Constants) ---
# NOTE: Changed back to the original request: 'all-mpnet-base-v2'.
CHUNK_SIZE_WORDS = 45
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' 

def chunk_text(text, chunk_size, document_id, start_page=1):
    """
    Splits the input text into chunks of specified word size.
    
    IMPORTANT: Column names are standardized here to match the evaluation script:
    'chunk_id', 'page_num', and 'chunk_txt'.
    """
    words = [word for word in re.split(r'\s+', text) if word]
    chunks_data = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)
        
        # Standardizing column names for consistency with the downstream evaluation script
        chunk_index = int(i / chunk_size)
        
        chunks_data.append({
            'chunk_id': f"{Path(document_id).stem}_{chunk_index}", # Unique ID combining document name and index
            'page_num': start_page,                              # Matches evaluation script
            'word_count': len(chunk_words),
            'chunk_txt': chunk_content                           # Matches evaluation script
        })
    
    return chunks_data

def extract_from_pdf(file_path):
    """Extracts text page-by-page from a PDF file using pypdf."""
    all_chunks = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks = chunk_text(page_text, CHUNK_SIZE_WORDS, file_path.name, start_page=i + 1)
                all_chunks.extend(chunks)
        print(f"✅ Extracted content from PDF: {file_path.name} ({len(reader.pages)} pages)")
    except Exception as e:
        print(f"❌ Error processing PDF {file_path.name}: {e}")
        
    return all_chunks

def extract_from_docx(file_path):
    """Extracts text from a DOCX file using python-docx."""
    all_chunks = []
    try:
        doc = Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        
        # NOTE ON PAGE NUMBERS: DOCX files are flow documents. Page number is assigned 1.
        chunks = chunk_text(full_text, CHUNK_SIZE_WORDS, file_path.name, start_page=1)
        all_chunks.extend(chunks)

        print(f"✅ Extracted content from DOCX: {file_path.name} (Assumed Page 1)")
    except Exception as e:
        print(f"❌ Error processing DOCX {file_path.name}: {e}")
        
    return all_chunks

def generate_embeddings(df, npy_output_path, model):
    """Generates embeddings for 'chunk_txt', saves them as .npy, or loads existing file."""
    if npy_output_path.exists():
        print(f"   --> Found existing embedding file: {npy_output_path.name}. Loading...")
        try:
            return np.load(npy_output_path)
        except Exception as e:
            print(f"   --> ⚠️ Could not load existing NPY file: {e}. Attempting regeneration.")

    try:
        # Use 'chunk_txt' column which is now standardized
        texts = df['chunk_txt'].tolist() 
        
        print(f"   --> Generating {len(texts)} embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Save the embeddings
        np.save(npy_output_path, embeddings)
        
        print(f"   --> 🎉 Saved embeddings to: {npy_output_path.name} (Shape: {embeddings.shape})")
        return embeddings
    except Exception as e:
        print(f"   --> ❌ Error generating embeddings for {npy_output_path.name}: {e}")
        return None

def create_faiss_index(embeddings, faiss_output_path):
    """Creates a FAISS index from the embeddings and saves it to a .faiss file, or skips if existing."""
    if faiss_output_path.exists():
        print(f"   --> Found existing FAISS index: {faiss_output_path.name}. Skipping index creation.")
        try:
            index = faiss.read_index(str(faiss_output_path))
            return index.ntotal
        except Exception as e:
            print(f"   --> ⚠️ Could not load existing FAISS index: {e}. Attempting regeneration.")

    if embeddings is None or len(embeddings) == 0:
        return 0

    try:
        # FAISS requires float32 data type
        vectors = embeddings.astype('float32')
        d = vectors.shape[1] # Dimension of the embedding vectors
        
        # Using IndexFlatL2 for simple Euclidean distance exact nearest neighbor search
        index = faiss.IndexFlatL2(d) 
        index.add(vectors) 

        # Save the FAISS index
        faiss.write_index(index, str(faiss_output_path))
        
        print(f"   --> 💾 Saved FAISS index to: {faiss_output_path.name} (Total vectors: {index.ntotal})")
        return index.ntotal
    except Exception as e:
        print(f"   --> ❌ Error creating FAISS index for {faiss_output_path.name}: {e}")
        return 0


def main():
    """Main function to scan the directory, process files, and write one Parquet/Numpy/FAISS file per input document, using checkpoints."""
    
    # Get user input for the directory path
    input_dir_str = input("Please enter the full path to the folder containing your .pdf and .docx files: ")
    INPUT_DIR = Path(input_dir_str)
    
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found at {INPUT_DIR}. Please check the path and try again.")
        return
        
    # --- Define and create output directories to match the RAG evaluation script ---
    TEXT_DIR = INPUT_DIR / "RAG_Data_Text"           # New folder for Parquet files
    EMBEDDINGS_DIR = INPUT_DIR / "RAG_Data_Embeddings" # Renamed folder for Numpy files
    FAISS_DIR = INPUT_DIR / "RAG_FAISS_Indices"      # Existing folder for FAISS files

    # Create the output directories if they don't exist
    TEXT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    FAISS_DIR.mkdir(exist_ok=True) 
    
    # Pre-load the Sentence Transformer model once (efficiency gain)
    try:
        print(f"Pre-loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load Sentence Transformer model. Please check installation and network connection. {e}")
        return
    
    print(f"\nStarting document processing in: {INPUT_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE_WORDS} words.")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}.\n")
    
    total_files_processed = 0
    total_chunks_written = 0
    total_embeddings_written = 0
    total_faiss_indexed = 0
    
    # Iterate over all files in the directory
    for file_path in INPUT_DIR.iterdir():
        if not file_path.is_file():
            continue
            
        # Define all output paths
        parquet_output_name = file_path.stem + ".parquet"
        parquet_output_path = TEXT_DIR / parquet_output_name # Parquet now goes into TEXT_DIR
        
        npy_output_name = file_path.stem + ".npy"
        npy_output_path = EMBEDDINGS_DIR / npy_output_name
        
        faiss_output_name = file_path.stem + ".faiss"
        faiss_output_path = FAISS_DIR / faiss_output_name 
        
        
        # Check if the file is one of our target input formats
        is_target_file = file_path.suffix.lower() in ['.pdf', '.docx']
        
        if not is_target_file and not parquet_output_path.exists():
            # Skip non-target files that aren't already generated outputs
            continue
            
        # --- 1. Parquet Checkpoint (Load or Create) ---
        df = None
        if parquet_output_path.exists():
            print(f"\nFound existing Parquet file: {parquet_output_path.name}. Loading...")
            try:
                df = pd.read_parquet(parquet_output_path)
                total_files_processed += 1
                total_chunks_written += len(df)
            except Exception as e:
                print(f"   --> ❌ Error loading Parquet file. Continuing with regeneration attempt. {e}")
                if is_target_file:
                    df = None
                else:
                    continue

        # If Parquet doesn't exist or failed to load, extract text and create it
        if df is None:
            if file_path.suffix.lower() == '.pdf':
                chunks = extract_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                chunks = extract_from_docx(file_path)
            else: 
                continue

            if chunks:
                df = pd.DataFrame(chunks)
                try:
                    df.to_parquet(parquet_output_path, index=False)
                    print(f"   --> 🎉 Wrote {len(df)} chunks to: {parquet_output_path.name}")
                    total_files_processed += 1
                    total_chunks_written += len(df)
                except Exception as e:
                    print(f"   --> ❌ Error writing Parquet file for {file_path.name}: {e}")
                    df = None
        
        if df is not None:
            # --- 2. Embeddings Checkpoint (Load or Generate) ---
            embeddings = generate_embeddings(df, npy_output_path, model)
            
            if embeddings is not None:
                total_embeddings_written += len(embeddings)
                
                # --- 3. FAISS Checkpoint (Skip or Create) ---
                faiss_count = create_faiss_index(embeddings, faiss_output_path)
                total_faiss_indexed += faiss_count
    
    print("\n" + "="*60)
    if total_files_processed > 0:
        print(f"Processing complete.")
        print(f"Total documents handled (loaded or processed): {total_files_processed}")
        print(f"Total chunks written/loaded: {total_chunks_written}")
        print(f"Total embeddings written/loaded: {total_embeddings_written}")
        print(f"Total FAISS vectors indexed/loaded: {total_faiss_indexed}")
        print(f"Output text (.parquet) files are in the '{{TEXT_DIR.name}}' subdirectory.")
        print(f"Output embedding (.npy) files are in the '{{EMBEDDINGS_DIR.name}}' subdirectory.")
        print(f"Output FAISS index (.faiss) files are in the '{{FAISS_DIR.name}}' subdirectory.")
    else:
        print("No supported files found or processed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()
