import os
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import traceback

# === Load tokenizer and LoRA-adapted model ===
model_path = r"/home/shrey/Desktop/Resume_Parser/resume-parser-mistral/mistral_resume_parser_model"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text_from_pdf(file_path):
    """Enhanced PDF text extraction that preserves layout information better"""
    full_text = ""
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
            if page_text:
                full_text += f"--- PAGE {page_num+1} ---\n{page_text}\n"
    
    return full_text

def extract_text_from_docx(file_path):
    """Enhanced DOCX text extraction that preserves structure better"""
    doc = Document(file_path)
    paragraphs_text = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs_text)

def extract_text(file_path):
    """Extract text from supported file formats"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx']:
        return extract_text_from_docx(file_path)
    elif ext in ['.txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print(f"⚠️ Error extracting text from image {file_path}: {e}")
        return ""

def process_chunk(text):
    """Process a single chunk of text using the Mistral model"""
    prompt = f"<|startoftext|>\nInput: {text}\nOutput:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
    
    outputs = model.generate(
        **input_ids,
        max_new_tokens=4096,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in result:
        json_part = result.split("Output:", 1)[1].split("<|endoftext|>")[0].strip()
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            return {"error": "Generated output is not valid JSON", "raw": json_part}
    return {"error": "Model did not return output"}

def extract_resume_details(text):
    """Extract structured data from resume text using the Mistral model"""
    return process_chunk(text)

# API endpoint function
@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        # Save the uploaded file
        file.save(file_path)
        print(f"File saved to {file_path}")
        
        # Extract text from the file
        resume_text = extract_text(file_path)
        print(f"Text extracted, length: {len(resume_text)} characters")
        
        # Process the resume to extract structured data
        extracted_data = extract_resume_details(resume_text)
        print(f"Data extracted, response length: {len(extracted_data)} characters")
        
        # Validate and ensure all required fields exist
        validated_data = validate_resume_data(extracted_data)
        
        # Clean up the file
        os.remove(file_path)
        
        return jsonify(validated_data), 200
    
    except Exception as e:
        # Clean up the file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        print(f"Error processing resume: {str(e)}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        
        return jsonify({"error": str(e)}), 500
