import uvicorn
import os

if __name__ == "__main__":

    # even though uvicorn is running on 0.0.0.0 check 127.0.0.1 from the browser

    if "code" in os.getcwd():
        uvicorn.run("src.app.app:app", host="0.0.0.0", port=7001, log_level="debug",
                    proxy_headers=True, reload=False)
    else:
        # for running locally from IDE without docker
        uvicorn.run("src.app.app:app", host="0.0.0.0", port=7001, log_level="debug", proxy_headers=True, reload=False)
#curl -X 'POST' \'http://localhost:7001/predict/tf/' \ -H 'accept: application/json' \ -H 'Content-Type: application/json' \ -d '{"img_url": "https://www.safetysign.com/images/source/medium-images/Y1249J.jpg"}'
#Invoke-WebRequest -Uri 'http://localhost:7001/predict/tf/' ` -Method POST ` -Headers @{'accept' = 'application/json' 'Content-Type' = 'application/json'} `-Body '{"img_url": "https://www.safetysign.com/images/source/medium-images/Y1249J.jpg"}'