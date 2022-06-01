var getAltBalise = async (src) => {
  var myHeaders = new Headers();
  myHeaders.append("Content-Type", "application/json");

  var raw = JSON.stringify({
    file: src,
  });

  var requestOptions = {
    method: "POST",
    headers: myHeaders,
    body: raw,
    redirect: "follow",
  };
  fetch("http://192.168.1.25:5000/iadecode/from_url", requestOptions)
    .then((response) => console.log(response.text()))
    .then((result) => console.log(result))
    .catch((error) => console.log("error", error));
};

var images = document.getElementsByTagName("img");
for (var i = 0, l = images.length; i < l; i++) {
  //if alt is null or empty
  if (images[i].alt == "" || images[i].alt == null)
    images[i].alt = getAltBalise(images[i].src);
}
