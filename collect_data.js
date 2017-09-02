// Go to http://s.cafef.vn/Lich-su-giao-dich-VNINDEX-1.chn in browsers
// F12, move to tab Console. Paste this script -> Enter

// Create text area

$('body').append(
  '<div style="display: none;">' +
  '<textarea id="textbox"></textarea> ' +
  '<button id="create">Create file</button> '+
  '<a download="data_stock_market.csv" id="downloadlink" style="display: none">Download</a>' +
  '</div>'
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

  var create = $('#create');
  var textbox = $('#textbox');
  create.click(function() {
    var link = $('#downloadlink');
    link.attr('href', makeTextFile(textbox.val()));
    // $('#downloadlink').show();
    $('#downloadlink')[0].click();
  });
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
    var high_price = $($(rows_datas[i]).find('.Item_Price10')[5]).html().replace('&nbsp;', '');
    var low_price = $($(rows_datas[i]).find('.Item_Price10')[6]).html().replace('&nbsp;', '');
    close_price = close_price.replace(',', '.');
    high_price = high_price.replace(',', '.');
    low_price = low_price.replace(',', '.');
    datas.push({
      date: date,
      close_price: close_price,
      high_price: high_price,
      low_price: low_price
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
    datas.forEach(function(data) {
      $('#textbox').val(data.date + '|' + data.close_price + '|' + data.high_price + '|' + data.low_price + '\r\n' + $('#textbox').val());
      // $('#textbox').val(data.date + '|' + data.close_price + '\r\n' + $('#textbox').val())
    });
    $('#textbox').val('date,close_price|high_price|low_price\r\n' + $('#textbox').val());
    $('#create').trigger('click');
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
