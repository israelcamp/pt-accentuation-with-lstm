import React from "react";
import "./App.css";

function TextHigh({ original, transformed }) {
  const orig_splited = original.split("");
  return (
    <div className="TextHigh">
      {transformed.split("").map((v, i) =>
        v === orig_splited[i] ? (
          <React.Fragment key={v + i}>
            <font color="white">{v}</font>
          </React.Fragment>
        ) : (
          <React.Fragment key={v + i}>
            <font color="#3e4444">{v}</font>
          </React.Fragment>
        )
      )}
    </div>
  );
}

export default TextHigh;
