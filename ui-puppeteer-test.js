const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async ()=>{
  const url = process.env.DEPLOY_URL || 'http://localhost:8000';
  console.log('Testing URL:', url);

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox','--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  page.setViewport({width:1280, height:800});

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 120000 });

  const samplePath = path.resolve(__dirname,'sample.jpg');
  if (!fs.existsSync(samplePath)){
    console.log('sample.jpg not found in tests/, test will take screenshot only.');
  }

  const input = await page.$('input[type=file]');
  if (!input) {
    console.error('file input not found on page');
    await page.screenshot({path: path.resolve(__dirname,'artifacts','no-input.png')});
    process.exit(2);
  }
  fs.mkdirSync(path.resolve(__dirname,'artifacts'), {recursive:true});
  await input.uploadFile(samplePath);

  await page.click('#start');
  await page.waitForTimeout(1000);
  try {
    await page.waitForFunction(()=>{
      const done = Array.from(document.querySelectorAll('small')).some(el => /Done|Done/i.test(el.innerText));
      const dl = document.querySelector('button#downloadZip, a#downloadZip');
      return done || !!dl;
    }, {timeout: 180000});
  } catch(e){
    console.error('timeout waiting for result', e.message);
    await page.screenshot({path: path.resolve(__dirname,'artifacts','timeout.png')});
    await browser.close();
    process.exit(3);
  }

  await page.screenshot({path: path.resolve(__dirname,'artifacts','final.png'), fullPage:true});
  const downloadHref = await page.$eval('#downloadZip', el => el.href || el.getAttribute('data-href') || '');
  if (downloadHref){
    console.log('Found download href:', downloadHref);
    const view = await page.goto(downloadHref, {timeout:120000});
    const buffer = await view.buffer();
    fs.writeFileSync(path.resolve(__dirname,'artifacts','result.zip'), buffer);
  } else {
    console.log('No download link found; artifacts include screenshot.');
  }

  await browser.close();
  console.log('UI test finished successfully.');
})();