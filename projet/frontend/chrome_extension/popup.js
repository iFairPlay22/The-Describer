function handleClick() {
    let checkbox = document.getElementById("slider");
    if(checkbox == null) return;
    if (checkbox.checked) {
        chrome.storage.sync.set({'describer_enabled': true});
    } else {
        chrome.storage.sync.set({'describer_enabled': false});
    }
    document.getElementById("refreshButton").style.display = "block";
}

function refreshPage(){
    chrome.tabs.query({active: true, currentWindow: true}, function (arrayOfTabs) {
        chrome.tabs.reload(arrayOfTabs[0].id);
        window.location.reload();
    });
}

// If the extension is enabled, the checkbox is checked
if(chrome.storage.sync.get('describer_enabled', function(data) {
    if(data.describer_enabled) {
        document.getElementById('slider').checked = true;
    }
    else{
        document.getElementById('slider').checked = false;
    }
}));
let checkbox = document.getElementById('slider');
let refreshButton = document.getElementById('refreshButton');
checkbox.onclick = handleClick;
refreshButton.onclick = refreshPage;
