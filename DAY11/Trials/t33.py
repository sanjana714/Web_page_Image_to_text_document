# server python file
from flask import Flask, render_template, abort, safe_join, send_file
from pathlib import Path
import os
import datetime as dt

# create a server instance
app = Flask(__name__)
FolderPath = r"C:\Users\Nagasudhir\Downloads"

@app.route('/')

def getIconClassForFilename(fName):
    fileExt = Path(fName).suffix
    fileExt = fileExt[1:] if fileExt.startswith(".") else fileExt
    fileTypes = [ "jpg","jpeg","png"]
    fileIconClass = f"bi bi-filetype-{fileExt}" if fileExt in fileTypes else "bi bi-file-earmark"
    return fileIconClass

# route handler
@app.route('/reports/', defaults={'reqPath': ''})
@app.route('/reports/<path:reqPath>')
def getFiles(reqPath):
    # Join the base and the requested path
    # could have done os.path.join, but safe_join ensures that files are not fetched from parent folders of the base folder
    absPath = safe_join(FolderPath, reqPath)

    # Return 404 if path doesn't exist
    if not os.path.exists(absPath):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(absPath):
        return send_file(absPath)

    # Show directory contents
    def fObjFromScan(x):
        fileStat = x.stat()
        # return file information for rendering
        return {'name': x.name,
                'fIcon': "bi bi-folder-fill" if os.path.isdir(x.path) else getIconClassForFilename(x.name),
                'relPath': os.path.relpath(x.path, FolderPath).replace("\\", "/"),
               }
    fileObjs = [fObjFromScan(x) for x in os.scandir(absPath)]
    # get parent directory url
    parentFolderPath = os.path.relpath(
        Path(absPath).parents[0], FolderPath).replace("\\", "/")
    return render_template('files.html.j2', data={'files': fileObjs,
                                                 'parentFolder': parentFolderPath})

# run the server
app.run(host="0.0.0.0", port=50100, debug=True)