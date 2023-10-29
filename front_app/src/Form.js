import React, { useState, useEffect } from 'react';
import api from './api';

function Form() {
    const [inputString, setInputString] = useState('');
    const [result, setResult] = useState('');

    const [info, setInfo] = useState("");

    const handleInputChange = (e) => {
        setInputString(e.target.value);
    };

    const sendData = () => {
        fetch('/convert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ input_string: inputString }),
        })
        .then((response) => response.json())
        .then((data) => {
            if ('predict' in data) {
                setResult(`Prediction: ${data.predict}`);
            } else if ('error' in data) {
                setResult(`Error: ${data.error}`);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    };

    const fetch_rta = async () => {
        const response = await api.post("/convert", {"input_string": inputString });
        setInfo(response.data);
        
      }

    useEffect(() => {
        fetch_rta().then((res) => {
          setInfo(res);
        });
      }, []);

      

    const sendDatav2 = () => {
          
    }        

    return (
        <div>
            <h1>React Component for FastAPI</h1>
            <input
                type="text"
                value={inputString}
                onChange={handleInputChange}
                placeholder="Enter an input string"
            />
            <button onClick={fetch_rta}>Send Data</button>
            <p>{info}</p>
        </div>
    );
}

export default Form;
