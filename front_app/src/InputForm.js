import React, { useState } from "react";
import api from "./api";

function InputForm() {
  const [inputString, setInputString] = useState("");
  const [result, setResult] = useState("");

  const handleSubmit = async () => {
    try {
        const response = await api.post("/convert", { "input_string": inputString });
        setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <h1>A que ODS pertenece?</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter input string"
          value={inputString}
          onChange={(e) => setInputString(e.target.value)}
        />
        <button type="submit" onClick={handleSubmit}>Submit</button>
      </form>
      <p>{result}</p>
    </div>
  );
}

export default InputForm;
