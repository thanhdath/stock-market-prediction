// Go to http://s.cafef.vn/Lich-su-giao-dich-VNINDEX-1.chn in browsers
// F12, move to tab Console. Paste this script
// After load to page 100, look at bottom, click "Create new file" and click 'Download'

// Create text area

$('body').append(
  '<textarea id="textbox">Type something here</textarea> ' +
  '<button id="create">Create file</button> '+
  '<a download="data_stock_market.txt" id="downloadlink" style="display: none">Download</a>'
);

// Create downloadable

(function () {
var textFile = null,
  makeTextFile = function (text) {
    text = text.replace(/\n/g, "\r\n");
    var data = new Blob([text], {type: 'text/plain'});

    // If we are replacing a previously generated file we need to
    // manually revoke the object URL to avoid memory leaks.
    if (textFile !== null) {
      window.URL.revokeObjectURL(textFile);
    }

    textFile = window.URL.createObjectURL(data);

    return textFile;
  };


  var create = document.getElementById('create'),
    textbox = document.getElementById('textbox');

  create.addEventListener('click', function () {
    var link = document.getElementById('downloadlink');
    link.href = makeTextFile(textbox.value);
    link.style.display = 'block';
  }, false);
})();

// Collect datas

var page = 1;
var datas = [];

function collectDatasInCurrentPage() {
  // Collect data in current page
  var table = $('#divStart').find('table#GirdTable2');
  var rows = table.find('tbody > tr');
  // rows datas in elements 2 .. 21
  var rows_datas = [];
  for (var i = 2; i <= 21; i++) {
    rows_datas.push(rows[i]);
  }

  for (var i = 0; i < rows_datas.length; i++) {
    var date = $($(rows_datas[i]).find('.Item_DateItem')[0]).html();
    var close_price = $($(rows_datas[i]).find('.Item_Price10')[0]).html().replace('&nbsp;', '');
    datas.push({
      date: date,
      close_price: close_price
    })
  }
}

function loadPage() {
  __doPostBack('ctl00$ContentPlaceHolder1$ctl03$pager2', '' + page);
}

function collect() {
  collectDatasInCurrentPage();
  if (page >= 100) {
    // Print datas in textarea
    $('#textbox').val('date, close_price\r\n');
    datas.forEach(function(data) {
      $('#textbox').val($('#textbox').val() + data.date + ',' + data.close_price + '\r\n');
    });
    return;
  }
  page++;
  loadPage();
  var check = setInterval(function() {
    var table = $('#divStart').find('table.CafeF_Paging');
    var button_pages = table.find('tbody > tr > td');
    var load_done = false;
    button_pages.toArray().forEach(function(button) {
      if (parseInt($(button).find('span > strong').html()) == page) {
        load_done = true;
      }
    });
    if(load_done) {
      clearInterval(check);
      collect();
    }
  }, 100);
}

collect();
