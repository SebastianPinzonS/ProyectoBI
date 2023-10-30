import React, { useState } from "react";
import axios from "axios";
import "./InputForm.css"

function InputForm() {
  const [inputString, setInputString] = useState("");
  const [result, setResult] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/convert", { "input_string": inputString });
      setResult(response.data.predict);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleInputChange = ((e) => {
		const newName = e.target.value;
		setInputString(newName);
	});


  return (
    <div className="centered-container">
      <h1>Ingresa el texto al que le quieres realizar la prediccion de a cual ODS corresponde</h1>
      <form >
        <input
          type="text"
          placeholder="Enter input string"
          value={inputString}
          onChange={handleInputChange}
          className="custom-input"
        />
        <button onClick={handleSubmit}>Submit</button>
      </form>
      <p className="result-text">Result: {result}</p>
    </div>
  );
}

export default InputForm;
