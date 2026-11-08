Run this isolated frontend without touching your existing detector files.

1. Open a terminal in this folder.
2. Start the app with:
   C:/Python313/python.exe classical/app.py
3. Open http://127.0.0.1:5000
4. Login with admin / mudra123

Notes:
- Live camera works with no JavaScript because Flask/OpenCV reads the local webcam and streams it into the page.
- Upload prediction uses a normal HTML form.
- The model file is reused from ../mudra_model.keras.
