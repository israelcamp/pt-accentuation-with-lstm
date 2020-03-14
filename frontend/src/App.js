import React, { useState } from "react";
import TextHighlighter from "./text-high";
import "./App.css";

function sendWebRequest(url, method, objectToSend, headers = null) {
  return fetch(url, {
    method: "POST", // *GET, POST, PUT, DELETE, etc.
    // headers: {
    //   ContentType: "application/json"
    // },
    body: objectToSend
  })
    .then(async response => {
      if (response !== undefined && response !== null && response.ok) {
        const parsedJson = await response.json();
        return { Message: "OK", result: parsedJson };
      } else {
        return { Message: "error", result: response };
      }
    })
    .catch(error => {
      return { Message: "error", result: error };
    });
}

function App() {
  const [original, setOriginal] = useState("o apartamento nao pegou fogo.");
  const [transformed, setTransformed] = useState(
    "O apartamento não pegou fogo."
  );

  const onSend = () => {
    if (original.length > 5) {
      const data = new FormData();
      data.append("text", original);
      const resp = sendWebRequest("http://127.0.0.1:5000/", "POST", data);
      resp.then(r => setTransformed(r.result.predicted_text));
    }
  };

  return (
    <div className="App">
      <div className="TextBox">
        <textarea
          className="InputArea"
          onChange={e => setOriginal(e.target.value)}
          value={original}
        ></textarea>
        <button type="button" onClick={onSend} disabled={original.length > 5}>
          ENVIAR
        </button>
        {transformed === null ? (
          <></>
        ) : (
          <TextHighlighter original={original} transformed={transformed} />
        )}
      </div>
    </div>
  );
}

export default App;
