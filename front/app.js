import axios from 'axios';


function App() {
  const [info, setInfo] = useState("");

  const api = axios.create({
    baseURL: 'http://localhost:8000'
});


  const inputData = document.getElementById("inputData").value;

  const fetch_rta = async () => {
    const response = await api.post("/convert", { inputData });
    return response.data;
  }

  useEffect(() => {
    fetch_rta().then((res) => {
      setInfo(res);
    });
  }, []);

  return (
    <div>
        <h1>A que ODS pertenece?</h1>
        <textarea id="inputData" placeholder="Enter input data"></textarea>
        <button id="submitButton">Submit</button>
        <p id="result">{info}</p>
    </div>

  );
}
export default App;