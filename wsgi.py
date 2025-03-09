from app import app
import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    
    # Run app
    app.run(
        host='0.0.0.0',  # Make the server publicly available
        port=port,
        debug=False      # Disable debug mode in production
    )