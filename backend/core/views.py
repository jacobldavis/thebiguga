from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
import io
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path to import main.py and mfcc.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import compute_spectral_hash
from mfcc import compare_files

# Directory where voice prints are stored
VOICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABOUT_PATH = PROJECT_ROOT / 'frontend' / 'ABOUT.md'


@require_GET
def about_content(request):
    try:
        with ABOUT_PATH.open(encoding='utf-8') as fp:
            markdown = fp.read()
    except FileNotFoundError:
        return JsonResponse({'error': 'about file not found'}, status=404)
    except OSError as exc:
        return JsonResponse({'error': str(exc)}, status=500)

    return JsonResponse({'content': markdown})


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
            },
            '/api/register-voice/': {
                'method': 'POST',
                'description': 'Save a voice print WAV for a username',
                'accepts': 'multipart/form-data (audio + username)',
                'returns': 'JSON confirmation'
            },
            '/api/authenticate-voice/': {
                'method': 'POST',
                'description': 'Compare audio against a stored voice print',
                'accepts': 'multipart/form-data (audio + username)',
                'returns': 'JSON with similarity score (0-1)'
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


@csrf_exempt
def register_voice(request):
    """
    Save a voice print WAV file for a given username.
    Stores as backend/<username>.wav
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    username = request.POST.get('username', '').strip()
    if not username:
        return JsonResponse({'error': 'No username provided'}, status=400)

    try:
        wav_file = request.FILES['audio']
        save_path = os.path.join(VOICE_DIR, f'{username}.wav')
        with open(save_path, 'wb') as f:
            for chunk in wav_file.chunks():
                f.write(chunk)

        return JsonResponse({
            'status': 'ok',
            'username': username,
            'message': f'Voice print saved for {username}'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def authenticate_voice(request):
    """
    Compare uploaded audio against a stored voice print for a username.
    Uses compare_files() from mfcc.py.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    username = request.POST.get('username', '').strip()
    if not username:
        return JsonResponse({'error': 'No username provided'}, status=400)

    stored_path = os.path.join(VOICE_DIR, f'{username}.wav')
    if not os.path.exists(stored_path):
        return JsonResponse({
            'error': f'No voice print found for "{username}". Record one first.'
        }, status=404)

    try:
        # Save incoming audio to a temp file for comparison
        wav_file = request.FILES['audio']
        tmp_path = os.path.join(VOICE_DIR, f'_tmp_auth_{username}.wav')
        with open(tmp_path, 'wb') as f:
            for chunk in wav_file.chunks():
                f.write(chunk)

        # compare_files expects paths without .wav extension
        stored_name = os.path.join(VOICE_DIR, username)
        tmp_name = os.path.join(VOICE_DIR, f'_tmp_auth_{username}')

        similarity = compare_files(stored_name, tmp_name)

        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return JsonResponse({
            'similarity': float(similarity),
            'username': username
        })

    except Exception as e:
        # Clean up temp file on error
        tmp_path = os.path.join(VOICE_DIR, f'_tmp_auth_{username}.wav')
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return JsonResponse({'error': str(e)}, status=500)
