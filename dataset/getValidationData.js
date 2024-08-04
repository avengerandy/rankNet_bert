let outputData = []
let reviews = document.getElementsByClassName('text');
for (let index = 0; index < reviews.length; index++) {
    outputData.push(reviews[index].innerText)
}
let jsonFile = new Blob([JSON.stringify(outputData)], {type: 'application/json'});
let a = document.createElement('a');
a.href = URL.createObjectURL(jsonFile);
a.download = 'testdata';
a.click();
