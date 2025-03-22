import pymupdf
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def perform_ocr_from_pdf(pdf_path):
    
    doc = pymupdf.open(pdf_path)
    model_name = "microsoft/Florence-2-large"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load Florence-2 model and processor
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    prompt = "<OCR>"
    extracted_text = []
    
    for page_num in range(len(doc)):

        page = doc.load_page(page_num)
        
        # Convert PDF page to image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Prepare input for Florence-2
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device, torch_dtype)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
        
        # Decode and process OCR output
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(img.width, img.height))
        
        # Handle text output format
        if isinstance(parsed_answer, dict):
            extracted_text.append(parsed_answer.get("<OCR>", ""))
        else:
            extracted_text.append(parsed_answer)

    return "\n".join(extracted_text)
