import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const d = JSON.parse(readFileSync(new URL('../inputs/session.json', import.meta.url), 'utf8'));

test("像素及支持坐标同长", () => {
  assert.ok(d.blocks.every(b=>b.frames.every(f=>f.pix.length===b.positions.length)));
});

test("存在遮挡及低对齐置信", () => {
  assert.ok(d.blocks[0].frames.some(f=>f.quality<d.profile.alignQuality)&&d.blocks[0].frames.every(f=>f.pix.includes(null)));
});

test("重叠区及接近收缩门限的薄奇异方向", () => {
  assert.ok(d.blocks[0].positions[1][0]===d.blocks[1].positions[0][0]&&d.blocks[0].frames[0].pix[0]*d.blocks[0].frames[1].pix[1]-d.blocks[0].frames[1].pix[0]*d.blocks[0].frames[0].pix[1]<0.01);
});
