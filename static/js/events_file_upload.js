

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


input.on('fileuploaded', function(event, data, previewId, index) {
    var form = data.form, files = data.files, extra = data.extra,
        response = data.response, reader = data.reader;
    console.log('File uploaded triggered');
});