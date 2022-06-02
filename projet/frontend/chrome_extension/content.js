var getAltBalise = async (image) => {
  var myHeaders = new Headers();
  myHeaders.append("Content-Type", "application/json");
  let src = image.src
  var raw = JSON.stringify({
    file: src,
  });

  var requestOptions = {
    method: "POST",
    headers: myHeaders,
    body: raw,
    redirect: "follow",
  };
  
  return fetch("http://127.0.0.1:5000/iadecode/from_url", requestOptions)
    .then((response) => {
      return response.json();
    })
    .then((result) => {
      image.alt = result.message;
      return result;
    })
    .catch((error) => console.log("error", error));
};

var images = document.getElementsByTagName("img");
for (var i = 0, l = images.length; i < l; i++) {
  //if alt is null or empty
  if (images[i].alt == "" || images[i].alt == null){
    getAltBalise(images[i]);
  }
}
