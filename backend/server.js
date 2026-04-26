const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

app.post("/chat", (req, res) => {
  const message = req.body.message.toLowerCase();

  let reply = "Sorry, I didn't understand.";

  if (message.includes("hello")) reply = "Hello 👋";
  else if (message.includes("fees")) reply = "Fees details available";
  else if (message.includes("admission")) reply = "Admissions start in June";

  res.json({ reply });
});

app.listen(5000, () => {
  console.log("Server running on http://localhost:5000");
});