const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const htmlPath = path.join(__dirname, 'reduction', '00_naive', 'naive_reduction_visualization_tree.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const dom = new JSDOM(html, { runScripts: "dangerously", resources: "usable" });

dom.window.HTMLCanvasElement.prototype.getContext = () => ({
  clearRect:()=>{}, save:()=>{}, restore:()=>{}, beginPath:()=>{},
  moveTo:()=>{}, lineTo:()=>{}, stroke:()=>{}, fill:()=>{},
  fillText:()=>{}, fillRect:()=>{}, strokeRect:()=>{},
  roundRect:()=>{}, translate:()=>{}, scale:()=>{}
});

setTimeout(() => {
  try {
    const sel = dom.window.document.getElementById('select-algo');
    sel.value = 'shrinking';
    sel.dispatchEvent(new dom.window.Event('change'));
    
    const nextBtn = dom.window.document.getElementById('btn-next');
    for(let i=0; i<6; i++) {
        console.log('Progress:', dom.window.state.progress);
        nextBtn.click();
    }
    console.log('Final Progress:', dom.window.state.progress);
    console.log('SUCCESS');
  } catch(e) {
    console.error('ERROR:', e);
    process.exit(1);
  }
}, 500);
