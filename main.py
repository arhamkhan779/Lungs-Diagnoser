from flask import Flask, request, render_template
import os
import tempfile
from werkzeug.utils import secure_filename
from prescription_parser import PrescriptionParser

app = Flask(__name__)

# Replace this with your actual PrescriptionParser implementation



@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_text = None
    if request.method == 'POST':
        if 'prescriptions' not in request.files:
            extracted_text = "No files part"
        else:
            files = request.files.getlist('prescriptions')
            if not files or files[0].filename == '':
                extracted_text = "No files selected"
            else:
                temp_dir = tempfile.mkdtemp()
                saved_paths = []
                try:
                    for file in files:
                        filename = secure_filename(file.filename)
                        temp_path = os.path.join(temp_dir, filename)
                        file.save(temp_path)
                        saved_paths.append(temp_path)

                    parser = PrescriptionParser(saved_paths)
                    extracted_text = parser.process_all()
                finally:
                    # Cleanup temp files and folder
                    for path in saved_paths:
                        if os.path.exists(path):
                            os.remove(path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)

    return render_template('index.html', extracted_text=extracted_text)


if __name__ == '__main__':
    app.run(debug=True)
