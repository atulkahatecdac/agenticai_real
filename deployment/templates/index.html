<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LangGraph News Agent</title>
<style>
body { font-family: Arial,sans-serif; background:#f4f6f8; display:flex; flex-direction:column; align-items:center; padding:50px; }
h1 { color:#333; }
.input-container { display:flex; margin-top:20px; }
input[type="text"] { padding:10px; font-size:16px; width:300px; border-radius:5px; border:1px solid #ccc; }
button { padding:10px 20px; font-size:16px; margin-left:10px; border:none; border-radius:5px; background:#007bff; color:white; cursor:pointer; transition:.2s; }
button:hover { background:#0056b3; }
#resultBox { margin-top:30px; width:80%; max-width:800px; background:white; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); white-space:pre-wrap; max-height:400px; overflow-y:auto; font-size:15px; line-height:1.5; }
.spinner { border:4px solid #f3f3f3; border-top:4px solid #007bff; border-radius:50%; width:24px; height:24px; animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:10px; }
@keyframes spin { 0% { transform:rotate(0deg); } 100% { transform:rotate(360deg); } }
</style>
</head>
<body>
<h1>LangGraph News Agent</h1>
<div class="input-container">
  <input type="text" id="topicInput" placeholder="Enter topic..." />
  <button onclick="submitTopic()">Submit</button>
</div>
<div id="resultBox">Your report will appear here...</div>
<script>
async function submitTopic() {
  const topic = document.getElementById('topicInput').value.trim();
  const resultBox = document.getElementById('resultBox');
  if (!topic) { resultBox.textContent="Please enter a topic!"; return; }

  resultBox.textContent = "Fetching report... ";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  resultBox.appendChild(spinner);

  try {
    const response = await fetch('/run-agent', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({topic})
    });

    if (!response.ok) {
      const err = await response.json();
      resultBox.textContent = `Error: ${err.error || response.statusText}`;
      return;
    }

    spinner.remove();
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    resultBox.textContent = "";

    while(true) {
      const {value, done} = await reader.read();
      if(done) break;
      resultBox.textContent += decoder.decode(value, {stream:true});
      resultBox.scrollTop = resultBox.scrollHeight;
    }

  } catch(err) {
    resultBox.textContent = `Error: ${err.message}`;
  }
}
</script>
</body>
</html>
