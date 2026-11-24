import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import { crossField, crossSlot, crossRequest } from '../inputs/field-assets.mjs';

const data=JSON.parse(readFileSync(new URL('../inputs/field-assets.json',import.meta.url),'utf8'));
test('推荐域、可见轴和嵌入参数的格式与关联', () => {
  assert.equal(crossField(1).keys[0],crossField(383).keys[0]);
  assert.equal(crossSlot(91).length,data.dimension);
  assert.ok(data.pairs.some(([a,b])=>a===1&&b===19));
  assert.ok(data.triples.some(t=>t.slice(0,3).includes(1)));
  assert.ok(!data.policy[1][1].includes(1));
  assert.ok(data.slotFetchLimit<data.fields*data.batch);
  assert.ok(data.workBytes<data.fields**2*4);
  assert.ok(data.triples.some(t=>t.slice(0,3).every(x=>data.policy[1][1].includes(x))));
});
