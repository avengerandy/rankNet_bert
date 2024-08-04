const appIdPattern = new RegExp(/app\/([0-9]+)\//);
let path = window.location.pathname;
let appId = path.match(appIdPattern)[1];

const greatHelpfulPattern = new RegExp(/([0-9,]+) 個人認為這篇評論值得參考/);
const funnyHelpfulPattern = new RegExp(/([0-9,]+) 個人認為這篇評論很有趣/);
function GetDataByPage(appId, page) {
    let pageBlock = document.getElementById(`page${page}`);
    let apphubCards = pageBlock.getElementsByClassName('apphub_Card modalContentLink interactable');
    let outputData = [];
    for (let index = 0; index < apphubCards.length; index++) {
        let apphubCard = apphubCards[index];

        // helpful
        let helpfulBlock = apphubCard.getElementsByClassName('found_helpful')[0];
        let helpfulText = helpfulBlock.innerText;
        let greatHelpfulMatch = helpfulText.match(greatHelpfulPattern);
        if (!Array.isArray(greatHelpfulMatch)) {
            continue;
        }

        greatHelpfulCount = greatHelpfulMatch[1];
        greatHelpfulCount = parseInt(greatHelpfulCount.replaceAll(/,/g, ''), 10);

        funnyHelpfulCount = 0;
        let funnyHelpfulMatch = helpfulText.match(funnyHelpfulPattern);
        if (Array.isArray(funnyHelpfulMatch)) {
            funnyHelpfulCount = funnyHelpfulMatch[1];
            funnyHelpfulCount = parseInt(funnyHelpfulCount.replaceAll(/,/g, ''), 10);
        }

        // content
        let textContentBlock = apphubCard.getElementsByClassName('apphub_CardTextContent')[0];
        textContentBlock.removeChild(textContentBlock.getElementsByClassName('date_posted')[0]);
        let textContent = textContentBlock.innerText;
        textContent = textContent.replaceAll(/\n/g, ' ');

        outputData.push([appId, textContent, greatHelpfulCount, funnyHelpfulCount, (10 * (page - 1)) + index + 1]);
    }
    return outputData;
}

function download(outputData, fileName) {
    let jsonFile = new Blob([outputData], {type: 'application/json'});
    let a = document.createElement('a');
    a.href = URL.createObjectURL(jsonFile);
    a.download = fileName;
    a.click();
}

let lotal = document.getElementsByClassName('apphub_Card modalContentLink interactable').length;
let lotalPage = Math.ceil(lotal / 10);
let outputData = []
for (let page = 1; page <= lotalPage; page++) {
    pageOutputData = GetDataByPage(appId, page);
    outputData = outputData.concat(pageOutputData);
}

const symbolPatten = new RegExp(/([^\w\u4e00-\u9fa5])/g);
const catPatten = new RegExp(/(?:貓|猫).*(?:讚|赞).*摸/);
function score(item) {
    let longPlus = Math.log10(item[1].replaceAll(symbolPatten, '').length);
    let orderDecay = item[4] / (item[4] + 1);
    let catDecay = 1;
    let catMatch = item[1].match(catPatten);
    if (Array.isArray(catMatch)) {
        catDecay = 0.5;
    }
    let confidenceDecay = (item[3] + 1) / (item[3] + 2);
    return (item[2] / (item[3] + 1)) * longPlus * orderDecay * catDecay * confidenceDecay;
}

outputData.sort((a, b) => {
    let aScore = score(a);
    let bScore = score(b);
    if (aScore > bScore) {
        return -1;
    }
    if (aScore < bScore) {
        return 1;
    }
    return 0;
})

download(JSON.stringify(outputData), appId);
