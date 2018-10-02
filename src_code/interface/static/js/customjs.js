

    $("#calcdynamic").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/dynamic/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#dynamicReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/dynamic";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


    $("#calcdynamicwithouttraceback").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/dynamicwithouttraceback/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcdynamicwithouttracebackReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/dynamicwithouttraceback";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});

    $("#calcdynamicalternative").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/dynamicalternative/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcdynamicalternativeReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/dynamicalternative";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});



   $("#calcgreedy").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/greedy/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcgreedyReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/greedy";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


   $("#calcgreedyalternative").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/greedyalternative/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcgreedyalternativeReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/greedyalternative";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


//fdf
  $("#calcdivideandconquer").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/divideandconquer/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcdivideandconquerReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/divideandconquer";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});

//fdf
  $("#calcstripek").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/stripek/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcstripekReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/stripek";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


//fdf
  $("#calcrecursive").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/recursive/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcrecursiveReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/recursive";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


//fdf
  $("#calcbranchandboundwithpath").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/branchandboundwithpath/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcbranchandboundwithpathReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/branchandboundwithpath";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});


//fdf
  $("#calcbranchandboundwithqueue").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
  var str_A = $("#str_A").val();
  var str_B = $("#str_B").val();

  if(str_A == '' || str_B == ''){
    $("#errormsgdynamic").show();
   // alert("ffff");
    return false;
  }
    var url1 = "/branchandboundwithqueue/"+str_A+"/"+str_B;
    //console.log(url1);
  //this will redirect us in same window
   window.location = url1;
});

  $("#calcbranchandboundwithqueueReset").click(function(event){
    event.preventDefault();
  //this will find the selected website from the dropdown
    var url2 = "/branchandboundwithqueue";
    //console.log(url1);
  //this will redirect us in same window
   window.location = url2;
});
