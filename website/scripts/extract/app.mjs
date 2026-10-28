// Fatos do app lidos do próprio código: pacote, versões do Android, ABI, pastas de dados e o
// que o build liga ou desliga. Cada fato guarda o arquivo e a linha de onde veio; se o código
// mudar de forma e um padrão deixar de casar, o problema aparece no relatório (e para o build
// no modo estrito) em vez de o site seguir com um valor velho.
import fs from 'node:fs';
import path from 'node:path';

export const APP_SOURCES = [
  'app/build.gradle',
  '.github/workflows/XenDroid.yml',
  'emulator-core/src/main/java/xendroid/compose/Application.java',
  'app/src/main/java/xendroid/compose/settings/ConfigStore.kt',
  'app/src/main/java/xendroid/compose/driver/DriverSources.kt',
];

/** Nível da API do Android → versão (developer.android.com/tools/releases/platforms). */
const ANDROID = { 29: '10', 30: '11', 31: '12', 32: '12L', 33: '13', 34: '14', 35: '15', 36: '16' };

function finder(root, problems) {
  const cache = new Map();
  const read = file => {
    if (!cache.has(file)) cache.set(file, fs.readFileSync(path.join(root, file), 'utf8'));
    return cache.get(file);
  };
  /** Primeiro grupo do padrão no arquivo, com a linha; `null` e um problema se não casar. */
  return (file, re, what) => {
    const src = read(file);
    const m = re.exec(src);
    if (!m) {
      problems.push({ level: 'error', msg: `${file}: não achei ${what} (padrão ${re})` });
      return null;
    }
    const line = src.slice(0, m.index).split('\n').length;
    return { value: m[1], source: `${file}:${line}` };
  };
}

export function extractApp(root) {
  const problems = [];
  const find = finder(root, problems);
  const g = 'app/build.gradle';
  const appId = find(g, /^\s*applicationId\s+'([\w.]+)'/m, 'applicationId');
  const releaseSuffix = find(g, /applicationIdSuffix\s+project\.findProperty\('xendroidReleaseIdSuffix'\)\s*\?:\s*'([\w.]+)'/, 'o sufixo do pacote da release');
  const minSdk = find(g, /^\s*minSdk\s+(\d+)/m, 'minSdk');
  const targetSdk = find(g, /^\s*targetSdk\s+(\d+)/m, 'targetSdk');
  const abi = find(g, /abiFilters\s+'([\w-]+)'/, 'abiFilters');
  const updateRepo = find(g, /xendroidUpdateRepo'\)\s*\?:\s*'([\w./-]+)'/, 'o repositório padrão das atualizações');
  const community = find(g, /xendroidCommunityUrl'\)\s*\?:\s*'([^']*)'/, 'a URL padrão da comunidade');
  const catalog = find(g, /xendroidCatalogUrl'\)\s*\?:\s*'([^']*)'/, 'a URL padrão do catálogo');
  const patches = find(g, /from\("\$rootDir\/(patches\/[\w./-]+)"\)\s*\{\s*include\s+'\*\.patch\.toml'/, 'a cópia dos patches para o APK');
  const testSuffix = find('.github/workflows/XenDroid.yml', /-PxendroidReleaseIdSuffix=(\.[\w.]+)/, 'o sufixo do pacote de teste do CI');
  const a = 'emulator-core/src/main/java/xendroid/compose/Application.java';
  const dataDir = find(a, /getExternalFilesDir\("(\w+)"\)/, 'a pasta de dados do app');
  const globalConfig = find(a, /get_global_config_file\(\)\s*\{\s*return new File\(Application\.get_app_data_dir\(\),"([\w.-]+)"\)/, 'o arquivo da config global');
  const driverSource = find('app/src/main/java/xendroid/compose/driver/DriverSources.kt', /const val DEFAULT = "([\w./-]+)"/, 'a fonte padrão de drivers');
  const driverMax = find('app/src/main/java/xendroid/compose/driver/DriverSources.kt', /const val MAX = (\d+)/, 'o limite de fontes de drivers');
  const gameConfig = find('app/src/main/java/xendroid/compose/settings/ConfigStore.kt', /fun perGameConfigDir\(\): File = File\(globalConfigFile\(\)\.parentFile, "(\w+)"\)[\s\S]*?File\(perGameConfigDir\(\), "\$\{titleId\.uppercase\(\)\}\.config\.toml"\)/, 'o caminho da config por jogo');

  const v = f => (f ? f.value : null);
  const pkg = v(appId) && v(releaseSuffix) ? v(appId) + v(releaseSuffix) : null;
  const testPkg = v(appId) && v(testSuffix) ? v(appId) + v(testSuffix) : null;
  const min = Number(v(minSdk));
  const target = Number(v(targetSdk));
  if (minSdk && !ANDROID[min]) problems.push({ level: 'error', msg: `minSdk ${min} sem versão do Android conhecida; atualize a tabela em scripts/extract/app.mjs` });
  if (targetSdk && !ANDROID[target]) problems.push({ level: 'warn', msg: `targetSdk ${target} sem versão do Android conhecida` });
  const dataRoot = pkg && v(dataDir) ? `Android/data/${pkg}/files/${v(dataDir)}` : null;
  const gameDir = gameConfig ? /"(\w+)"/.exec(gameConfig.value)?.[1] || gameConfig.value : null;

  return {
    package: pkg,
    testPackage: testPkg,
    minSdk: min,
    minAndroid: ANDROID[min] || null,
    targetSdk: target,
    targetAndroid: ANDROID[target] || null,
    abi: v(abi),
    updateRepo: v(updateRepo),
    communityUrl: v(community),
    catalogUrl: v(catalog),
    patchesDir: v(patches),
    driverSource: v(driverSource),
    driverSourcesMax: Number(v(driverMax)),
    dataRoot,
    globalConfig: dataRoot && v(globalConfig) ? `${dataRoot}/${v(globalConfig)}` : null,
    gameConfigDir: dataRoot && gameDir ? `${dataRoot}/${gameDir}` : null,
    gameConfigPattern: dataRoot && gameDir ? `${dataRoot}/${gameDir}/<TITLE ID>.config.toml` : null,
    sources: Object.fromEntries(Object.entries({ appId, releaseSuffix, minSdk, targetSdk, abi, updateRepo, community, catalog, patches, testSuffix, dataDir, globalConfig, gameConfig, driverSource, driverMax })
      .filter(([, f]) => f).map(([k, f]) => [k, f.source])),
    problems,
  };
}
