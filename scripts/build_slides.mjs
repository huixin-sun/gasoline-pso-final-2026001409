// Artifact Tool authoring source. Called inside Artifact Session by populate().
// Numerical experiments and report reproduction do not depend on this runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
export async function populate(presentation, root) {
  const data=JSON.parse(await fs.readFile(path.join(root,'docs/slides_source.json'),'utf8'));
  presentation.slides.keep([]);
  const text=(slide,value,left,top,width,height,size,color='#253448',bold=false)=>{
    const box=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
    box.text=value;
    box.text.style={typeface:'SimHei',fontSize:size,color,bold};
  };
  for (const [i,item] of data.entries()) {
    const slide=presentation.slides.add({width:1280,height:720});
    slide.background.fill='#ffffff';
    text(slide,item.title,64,i===0?142:48,1152,95,i===0?54:43,'#176b89',true);
    if(i===0){
      item.body.forEach((line,j)=>text(slide,line,68,292+j*68,1144,60,j===0?30:27));
    }else if(item.figure){
      item.body.forEach((line,j)=>text(slide,line,64,142+j*45,1152,48,24));
      const y=item.body.length>2?292:247;
      slide.images.add({path:path.join(root,'figures',item.figure+'.png'),alt:item.title,fit:'contain',position:{left:75,top:y,width:1130,height:602-y}});
      text(slide,item.takeaway,64,618,1152,48,24,'#176b89');
    }else{
      item.body.forEach((line,j)=>text(slide,line,64,155+j*112,1152,96,28));
    }
    text(slide,`孙慧鑫 2026001409    ${i+1} / ${data.length}`,64,680,1152,24,17,'#718394');
    slide.speakerNotes.text=item.notes+'\n来源：给定论文第5—8章、附件一与附件四；实验数据见 results/*.csv，配置见 config/experiment.json。';
  }
  return data.length;
}
