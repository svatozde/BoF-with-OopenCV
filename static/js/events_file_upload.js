

// language=JQuery-CSS
var input = $("#input-id")





input.fileinput({
    uploadAsync: false,
    uploadUrl: "/",
    showAjaxErrorDetails: false,
    allowedFileExtensions: ["jpg", "png", "jpeg"]
});


input.on('fileloaded', function(event, file, previewId, index, reader) {
    console.log("fileloaded");
});

input.on('filebatchuploadsuccess', function(event,response){
    images = ['result-line.png'];
    var x = document.getElementById("results");

    while (x.hasChildNodes()) {
        x.removeChild(x.firstChild);
    }

    for (index = 0; index < response.response.length; ++index) {
        console.log(response.response[index]);
        var imgTag = '<div class="col-lg-3 col-md-4 col-xs-6"><a href="#" class="d-block mb-4 h-100"><img class="img-fluid img-thumbnail" src="images/'+ response.response[index]+'" alt=""></a></div>';
        var div = document.createElement('div');
        div.innerHTML = imgTag;
        console.log(div.childNodes);
        console.log(div.childNodes[0]);
        x.appendChild(div.childNodes[0]);
    }
});


input.on('fileuploaded', function(event, data, previewId, index) {
    var form = data.form, files = data.files, extra = data.extra,
        response = data.response, reader = data.reader;
    console.log(response);

});