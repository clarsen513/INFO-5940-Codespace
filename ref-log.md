I asked google gemini about how to use the built in langchain loaders when the file was on a user's local file system
- I asked because the examples we were provided all used file paths, but the way a file is processed in the starter code did not create a file path for the uploaded file.
- It suggested using tempfiles to be able to pass a file path to the langchain loaders. This suggestion worked well after I figured out how to work with temp files by consulting python documentation.


GitHub copilot was active while I completed the assignment. This just suggested code completions and didn't provide implementation help but did speed up development by making my coding faster.