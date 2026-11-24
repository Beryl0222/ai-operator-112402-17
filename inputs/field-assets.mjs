import {readFileSync} from 'node:fs';
const data = JSON.parse(readFileSync(new URL('./field-assets.json', import.meta.url), 'utf8'));
export function crossField(id) {
  const alias=data.aliases.find(([a,b])=>a===id||b===id);
  return {id,keys:[alias?alias[2]:(73*id)%data.slots,(97*id+7)%data.slots,(113*id+13)%data.slots],weights:[.5,.375,.125]};
}
export function crossSlot(key) {
  return Float64Array.from({length:data.dimension},(_,i)=>Math.sin(.23*(key+13)*(i+1))*.25);
}
export function crossRequest(index) {
  const policy = data.policy[index%data.policy.length];
  return {exposure:policy[0],fields:policy[1],upstream:data.pairs.map((p,i)=>[p[0],p[1],.125*(i+1)])};
}
