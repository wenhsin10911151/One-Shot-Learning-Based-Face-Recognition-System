let video;

document.addEventListener('DOMContentLoaded', () => {
  console.log("DOM Loaded");
  video = document.getElementById('webcam');

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
    });

  document.getElementById('capture-btn').addEventListener('click', () => {
    console.log("Button clicked");

    const canvas = document.getElementById('captured-image'); // 使用先前在HTML中定義的canvas
    const loadingMessage = document.getElementById('loading-message');
    const predictionResult = document.getElementById("prediction-result");
    const captureBtn = document.getElementById('capture-btn');
    const resetBtn = document.getElementById('reset-btn');

    captureBtn.disabled = true; // 禁用capture按鈕

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    video.style.display = 'none'; // 隱藏video
    canvas.style.display = 'block'; // 顯示canvas
    loadingMessage.style.display = 'block'; // 顯示"預測中，請稍等"的訊息

    const image = canvas.toDataURL('image/jpg').replace(/^data:image\/\w+;base64,/, '');
    const fs = require('fs');
    fs.writeFile('photo.jpg', image, 'base64', function (err) {
      if (err) console.log(err);
    });

    // 按鈕被觸發以後，才會開始執行程式
    const { exec } = require('child_process');

    exec('python onePeople.py', (error, stdout, stderr) => {
      console.log('執行程式');
      if (error) {
        console.error(`Error: ${error}`);
        return;
      }

      const startMarker = "<!--START-->";
      const endMarker = "<!--END-->";
      const startIndex = stdout.indexOf(startMarker) + startMarker.length;
      const endIndex = stdout.indexOf(endMarker);

      const studentMapping = {
        0: "蘇志文教授",
        1: "田筱榮教授",
        2: "朱守禮教授",
        3: "吳宜鴻教授",
        4: "余執彰教授",
        5: "張元翔教授",
        6: "莊啓宏教授",
        7: "湯松年教授",
        8: "鄭維凱教授",
        9: "賴建宏教授",
        10: "鐘武君教授",
        11: "楊明豪教授",
        12: "夏延德教授",
        13: "林玟君同學",
        14: "徐文心同學"
      }
      
      if (startIndex !== -1 && endIndex !== -1) {
        const jsonStr = stdout.substring(startIndex, endIndex).trim();
        let result = JSON.parse(jsonStr);

        const studentName = studentMapping[result.face_prediction];
        if (studentName) {
          predictionResult.innerText = `${studentName}已簽到！`;
        } else {
          predictionResult.innerText = "Unknown ID, please try again!";
        }

        loadingMessage.style.display = 'none'; // 隱藏訊息
        canvas.style.display = 'none'; // 隱藏截圖的照片

        resetBtn.style.display = 'block'; // 顯示重置按鈕

      } else {
        console.error("Couldn't extract JSON from Python output.");
      }

      console.error(`stderr: ${stderr}`);
    });

  });
});

function displayResultOnWebPage(prediction) {

  document.getElementById("prediction-result").innerText = prediction;
}

// 加入重置按鈕的事件監聽器
document.getElementById('reset-btn').addEventListener('click', function() {
  const canvas = document.getElementById('captured-image');
  const predictionResult = document.getElementById("prediction-result");
  const captureBtn = document.getElementById('capture-btn');
  const resetBtn = document.getElementById('reset-btn');

  video.style.display = 'block'; // 重新顯示video
  canvas.style.display = 'none'; // 隱藏canvas
  predictionResult.innerText = ''; // 清空預測結果
  captureBtn.disabled = false; // 重新啟用capture按鈕
  resetBtn.style.display = 'none'; // 隱藏重置按鈕
});