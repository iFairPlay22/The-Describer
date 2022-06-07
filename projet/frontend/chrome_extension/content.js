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
  
  fetch("https://www.loicfournier.fr/iadecode/from_url", requestOptions)
    .then((response) => {
      if(response.status == 200){
        return response.json();
      }
      console.log(response)
      return null
    })
    .then((result) => {
      if(result == null){
        console.log("result null")
        return "";
      }
      let msg = result.message;
      image.alt = msg;
      if(image.title == "" || image.title == null) {
        image.title = msg;
      }
      return result;
    })
    .catch((error) => {
      console.log("erreur 400")
      console.log("error", error)}
    );
};

if(chrome.storage.sync.get('describer_enabled', function(data) {
  if(data.describer_enabled) {
    var images = document.getElementsByTagName("img");
    for (var i = 0, l = images.length; i < l; i++) {
      //if alt is null or empty
      if (images[i].alt == "" || images[i].alt == null){
        console.log(i);
        getAltBalise(images[i]);
      }
      console.log(i + images[i].alt);
    }
  }
}));