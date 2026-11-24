import {readFileSync} from 'node:fs';
import * as input from './inputs/field-assets.mjs';

const source=JSON.parse(readFileSync(new URL('./inputs/field-assets.json',import.meta.url),'utf8'));
console.log(JSON.stringify({名称:'推荐域、可见轴和嵌入参数',首项:source['fields'],字段:Object.keys(source),取数函数:Object.keys(input)},null,2));
