from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import sys
import os

# Add parent directory to path to import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import compute_spectral_hash


def api_info(request):
    """
    Root endpoint providing API information.
    """
    return JsonResponse({
        'name': 'Auditory Auth API',
        'version': '1.0',
        'description': 'Spectral fingerprint generation API',
        'endpoints': {
            '/api/process-audio/': {
                'method': 'POST',
                'description': 'Process WAV file and return spectral hash',
                'accepts': 'multipart/form-data (audio field)',
                'returns': 'JSON with hash and metadata'
            }
        },
        'frontend': 'http://localhost:3000',
        'note': 'This is an API-only backend. Please use the frontend application.'
    })


@csrf_exempt
def process_audio(request):
    """
    Endpoint to receive WAV file and return spectral hash.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)
    
    try:
        # Get the uploaded WAV file
        wav_file = request.FILES['audio']
        
        # Read file into BytesIO for processing
        wav_data = io.BytesIO(wav_file.read())
        
        # Compute spectral hash
        result = compute_spectral_hash(wav_data)
        
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
