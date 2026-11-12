#!/usr/bin/env node
// npm install playwright-core; PLAYWRIGHT_MODULE may name an existing index.mjs.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const [url,output,routeFile]=process.argv.slice(2);
if(!url || !output) throw Error('usage: capture.mjs URL OUTPUT.mp4 [floor-path.json]');
const seconds=Number(process.env.SECONDS || 60), width=Number(process.env.WIDTH || 1280),height=Number(process.env.HEIGHT || 720);
const target=path.resolve(output);fs.mkdirSync(path.dirname(target),{recursive:true});
const browser=await chromium.launch({executablePath:process.env.CHROME || (process.platform==='darwin'?'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome':'/usr/bin/google-chrome'),headless:true,ignoreDefaultArgs:['--enable-unsafe-swiftshader'],args:['--ignore-gpu-blocklist','--disable-gpu-vsync','--disable-frame-rate-limit',...(process.platform==='darwin'?['--use-angle=metal']:['--use-angle=vulkan','--enable-features=Vulkan'])]});
process.once('SIGTERM',async()=>{await browser.close();process.exit(143);});
let stream;
try {
 const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>{errors.push(String(e));console.error('PAGE ERROR',String(e));});
 page.on('console',m=>{if(m.type()==='error')console.error('BROWSER',m.text());});
 await page.goto(url);
 await page.waitForFunction(()=>window.__viewer?.ready && window.__viewer.collide?.ready,null,{timeout:180000});
 await page.waitForTimeout(1800);
 const gpu=await page.evaluate(()=>window.__viewer.dbg().gpu);if(/swiftshader|llvmpipe|software/i.test(gpu)||!gpu)throw Error(`real GPU required: ${gpu}`);
 console.log('GPU',gpu);
 const input=routeFile?JSON.parse(fs.readFileSync(routeFile,'utf8')):null;
 const plan=await page.evaluate(input=>{
  const v=window.__viewer,c=v.three.camera,up=c.up.clone(),start=c.position.clone(),mpu=v.CFG.metersPerUnit || 1;
  // The viewer's own level basis (state.walkBasis); pages without it get the same construction (levelBasis).
  const ref=[[0,0,-1],[1,0,0],[0,1,0]].reduce((a,r)=>Math.abs(up.x*r[0]+up.y*r[1]+up.z*r[2])<Math.abs(up.x*a[0]+up.y*a[1]+up.z*a[2])-1e-9?r:a);
  const f=v.walkBasis?start.clone().set(...v.walkBasis.forward):start.clone().set(...ref).addScaledVector(up,-start.clone().set(...ref).dot(up)).normalize();
  const right=v.walkBasis?start.clone().set(...v.walkBasis.right):f.clone().cross(up).normalize();
  const eye=1.5/mpu,step=.15/mpu;
  const eyePoint=(x,z)=>{const p=start.clone().addScaledVector(right,x).addScaledVector(f,z);const grounded=v.projectFloor(p.toArray());return grounded?p.set(...grounded):null;};
  if(input) {const points=(input.points || input).map(p=>p.map((n,i)=>n+up.getComponent(i)*eye));return {points,source:'supplied floor positions',step};}
  // Search a finite planning window, never constrain navigation to it. Capture poses
  // provide scale/extent only; route choice prefers clear cells distant from them.
  const poses=v.CFG.path || [];
  const coords=poses.map(p=>{const q=start.clone().set(...p.p).sub(start);return [q.dot(right),q.dot(f)];});
  const radius=coords.length?Math.min(15/mpu,Math.max(...coords.flat().map(Math.abs))+1/mpu):Math.min(v.CFG.radius||5,10/mpu);
  const n=Math.ceil(2*radius/step)+1,mid=Math.floor(n/2),clear=new Uint8Array(n*n);
  for(let z=0;z<n;z++)for(let x=0;x<n;x++){const p=eyePoint((x-mid)*step,(z-mid)*step);clear[z*n+x]=p && v.collide.penetration(...p.toArray())===0?1:0;}
  const seed=mid*n+mid;if(!clear[seed])throw Error('spawn is occupied');
  const edgeCache=new Map();
  const neighbors=i=>{const x=i%n,z=Math.floor(i/n);return [[1,0],[-1,0],[0,1],[0,-1]].map(([a,b])=>[x+a,z+b]).filter(([a,b])=>a>=0&&b>=0&&a<n&&b<n).map(([a,b])=>b*n+a).filter(j=>{if(!clear[j])return false;const key=i+":"+j;if(!edgeCache.has(key)){const a=eyePoint((i%n-mid)*step,(Math.floor(i/n)-mid)*step),b=eyePoint((j%n-mid)*step,(Math.floor(j/n)-mid)*step);edgeCache.set(key,v.canWalk(a.toArray(),b.toArray()));}return edgeCache.get(key);});};
  const prev=new Int32Array(n*n).fill(-1),queue=[seed];prev[seed]=seed;
  for(let at=0;at<queue.length;at++)for(const j of neighbors(queue[at]))if(prev[j]<0){prev[j]=queue[at];queue.push(j);}
  if(queue.length<10)throw Error('less than 10 reachable floor cells: '+JSON.stringify({start:start.toArray(),floor:v.collide.floorHeight,clear:clear.reduce((a,b)=>a+b),debug:v.dbg()}));
  const coord=i=>[(i%n-mid)*step,(Math.floor(i/n)-mid)*step];
  const off=i=>{const [x,z]=coord(i);let best=Infinity;for(let j=1;j<coords.length;j++){const a=coords[j-1],b=coords[j],dx=b[0]-a[0],dz=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(z-a[1])*dz)/(dx*dx+dz*dz||1)));best=Math.min(best,Math.hypot(x-a[0]-t*dx,z-a[1]-t*dz));}return Number.isFinite(best)?best:0;};
  const far=queue.reduce((a,b)=>off(b)>off(a)?b:a,seed);
  const xs=queue.map(i=>coord(i)[0]),zs=queue.map(i=>coord(i)[1]),lo=[Math.min(...xs),Math.min(...zs)],hi=[Math.max(...xs),Math.max(...zs)];
  const nearest=t=>queue.reduce((a,b)=>{const d=i=>{const q=coord(i);return Math.hypot(q[0]-t[0],q[1]-t[1]);};return d(b)<d(a)?b:a;},seed);
  const center=nearest(lo.map((v,i)=>(v+hi[i])/2));
  const goals=[far,nearest(lo),center,nearest([hi[0],lo[1]]),nearest(hi),center,nearest([lo[0],hi[1]]),seed];
  let from=seed;const route=[seed];
  for(const goal of goals){const parent=new Int32Array(n*n).fill(-1),q=[from];parent[from]=from;for(let at=0;at<q.length&&parent[goal]<0;at++)for(const j of neighbors(q[at]))if(parent[j]<0){parent[j]=q[at];q.push(j);}const segment=[];for(let i=goal;i!==from;i=parent[i])segment.push(i);route.push(...segment.reverse());from=goal;}
  // Collapse collinear grid steps; preserve corners for collision-safe routing.
  const compact=route;
  return {points:compact.map(i=>{const [x,z]=coord(i);return eyePoint(x,z).toArray();}),source:'reachable grid: farthest from capture, middle, four corners',reachableCells:queue.length,maxPlannedOffPathM:off(far)*mpu,step};
 },input);
 fs.writeFileSync(target+'.path.json',JSON.stringify(plan,null,2));console.log('route',plan.points.length,plan.reachableCells);
 if(!process.env.NO_VIDEO){stream=fs.createWriteStream(target+'.webm');await page.exposeFunction('recordChunk',b64=>stream.write(Buffer.from(b64,'base64')));}
 const metrics=await page.evaluate(async({plan,seconds,video})=>{
  const v=window.__viewer,c=v.three.camera,renderer=v.three.renderer,mpu=v.CFG.metersPerUnit||1;
  const points=plan.points;let length=0;for(let i=1;i<points.length;i++)length+=Math.hypot(...points[i].map((x,j)=>x-points[i-1][j]));
  v.routeSpeed=Math.max(.2,Math.min(1.2,length*mpu/(seconds*.75)));v.routeTarget=points[1];
  const times=[],movingTimes=[],positions=[];let last=performance.now(),at=1,raf,draws=0,waypointsReached=0,completedLoops=0;
  const recordCanvas=document.createElement("canvas");recordCanvas.width=innerWidth;recordCanvas.height=innerHeight;
  const ctx=recordCanvas.getContext("2d");
  const original=renderer.render.bind(renderer);renderer.render=(...a)=>{draws++;const result=original(...a);if(video && !renderer.getRenderTarget())ctx.drawImage(renderer.domElement,0,0,recordCanvas.width,recordCanvas.height);return result;};
  const tick=now=>{times.push(now-last);if(v.isMoving?.())movingTimes.push(now-last);last=now;if(c.position.clone().set(...points[at]).sub(c.position).addScaledVector(c.up,-c.position.clone().set(...points[at]).sub(c.position).dot(c.up)).length()<.025/mpu){waypointsReached++;at=(at+1)%points.length;if(at===0)completedLoops++;v.routeTarget=points[at];}if(times.length%10===0)positions.push(c.position.toArray());raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);
  let rec,recordTimer,pending=Promise.resolve(),done;
  if(video){const stream=recordCanvas.captureStream(0);recordTimer=setInterval(()=>{ctx.drawImage(recordCanvas,0,0);stream.getVideoTracks()[0].requestFrame();},1000/30);rec=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp9',videoBitsPerSecond:6000000});rec.ondataavailable=e=>{pending=pending.then(async()=>{const bytes=new Uint8Array(await e.data.arrayBuffer());let s='';for(let i=0;i<bytes.length;i+=32768)s+=String.fromCharCode(...bytes.subarray(i,i+32768));await window.recordChunk(btoa(s));});};done=new Promise(r=>rec.onstop=r);rec.start(1000);}
  await new Promise(r=>setTimeout(r,seconds*1000));cancelAnimationFrame(raf);v.routeTarget=null;renderer.render=original;
  if(rec){clearInterval(recordTimer);rec.stop();await done;await pending;rec.stream.getTracks().forEach(t=>t.stop());}
  const elapsed=times.reduce((a,b)=>a+b,0),sorted=times.slice().sort((a,b)=>a-b);
  return {movingFps:movingTimes.length?movingTimes.length*1000/movingTimes.reduce((a,b)=>a+b,0):null,movingFrames:movingTimes.length,seconds:elapsed/1000,fps:times.length*1000/elapsed,p95FrameMs:sorted[Math.floor(sorted.length*.95)],drawCallsPerSecond:draws*1000/elapsed,positions,reachedWaypoint:at,waypointsReached,completedLoops,totalWaypoints:points.length,debug:v.dbg(),removedCollisionCells:v.collide.removedCells,splats:v.numSplats};
 },{plan,seconds,video:!process.env.NO_VIDEO});
 await page.screenshot({path:target+'.png'});
 fs.writeFileSync(target+'.json',JSON.stringify({gpu,width,height,recording:!process.env.NO_VIDEO,plan,metrics,errors},null,2));
 if(errors.length)throw Error(errors.join('\n'));
 console.log(JSON.stringify({fps:metrics.fps,p95FrameMs:metrics.p95FrameMs,reached:metrics.reachedWaypoint,total:metrics.totalWaypoints,position:metrics.debug.pos}));
 if(stream){await new Promise(r=>stream.end(r));stream=null;const ff=spawnSync('ffmpeg',['-y','-i',target+'.webm','-vf',`fps=30,tpad=stop_mode=clone:stop_duration=${seconds}`,'-t',String(seconds),'-c:v','libx264','-threads','4','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',target],{stdio:'inherit'});if(ff.status)throw Error('ffmpeg failed');}
} finally {stream?.end();await browser.close();}
