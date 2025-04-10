$(document).ready(function() {
    $('#fileup').change(function() {
        // Here, the code checks the file extension of the selected file
        var res = $('#fileup').val();
        var arr = res.split("\\");
        var filename = arr.slice(-1)[0];
        filextension = filename.split(".");
        filext = "." + filextension.slice(-1)[0];
        valid = [".jpg", ".png", ".jpeg", ".bmp"];
        // If the file extension is not in the valid list, it displays an error message
        if (valid.indexOf(filext.toLowerCase()) == -1) {
            $(".imgupload").hide("slow");
            $(".imgupload.ok").hide("slow");
            $(".imgupload.stop").show("slow");

            $('#namefile').css({ "color": "red", "font-weight": 700 });
            $('#namefile').html("File " + filename + " is not  pic!");

            $("#submitbtn").hide();
            $("#fakebtn").show();
        } else {
            // If the file extension is valid, it displays a success message
            $(".imgupload").hide("slow");
            $(".imgupload.stop").hide("slow");
            $(".imgupload.ok").show("slow");

            $('#namefile').css({ "color": "green", "font-weight": 700 });
            $('#namefile').html(filename);

            $("#submitbtn").show();
            $("#fakebtn").hide();
        }
    });
});
