# refstar_planner

小惑星観測前に、予定観測位置の矩形視野内に**測光較正に使える参照星が何個あるか**を確認する GUI ツール。

Pan-STARRS / Gaia DR3 / 2MASS (VizieR 経由) を主カタログとして使用する。

---

## インストール

```bash
pip install astropy astroquery pandas numpy matplotlib tkinter
```

---

## 起動方法

```bash
python3.12 refstar_gui.py
```

タブ切り替えで **固定モード** と **可変モード（小惑星）** を使い分ける。

---

## 固定モード（📍 固定モード）

任意の RA/Dec を指定して参照星を検索するモード。

### 入力項目

| 項目 | 説明 |
|------|------|
| RA / Dec | 赤経・赤緯（deg または hh:mm:ss / +dd:mm:ss） |
| 視野幅・高さ (arcmin) | 矩形視野のサイズ |
| PA (deg) | 位置角（North → East） |
| カタログ | `Pan-STARRS` / `Gaia` / `2MASS` / `SIMBAD` |
| バンド | 2MASS のみ: `J` / `H` / `Ks` を選択可 |
| 等級範囲 | 参照星として使う等級の下限・上限 |
| 等級誤差上限 | 参照星の等級誤差フィルタ |
| 最小離角 (arcsec) | 中心（小惑星位置など）からの最小距離 |

### 出力

- 参照星数と判定（GOOD / OK / MARGINAL / POOR / BAD）
- 視野プレビュー（星分布の散布図）
- 参照星リストの CSV 保存

---

## 可変モード（☄ 可変モード）

JPL Horizons から小惑星の時系列エフェメリスを取得し、各時刻の参照星数を評価するモード。

### 入力項目

| 項目 | 説明 |
|------|------|
| 小惑星名 | Horizons で検索可能な名前・番号（例: `Psyche`, `16`） |
| 観測地 | プリセットから選択または緯度・経度・標高を直接入力 |
| 観測開始・終了日時 | UTC |
| ステップ間隔 | エフェメリス取得間隔（例: `1h`, `30m`） |
| 視野・PA・カタログ・等級範囲 | 固定モードと同じ |

### 出力

- 時刻別の参照星数グラフ（航海薄明後の夜間帯をハイライト表示）
- 各時刻の視野プレビュー（スライダーで時刻を切り替え）
- サマリ表示（時刻・小惑星 RA/Dec・FoV 内星数・使用可能星数・判定）
- 参照星リストおよびサマリの CSV 保存

### 小惑星名が曖昧な場合

Horizons で複数候補が見つかった場合は、入力欄にドロップダウンで候補が表示される。候補を選択すると自動的に再検索する。

---

## 対応カタログ

### Pan-STARRS（推奨: 光学 r バンド測光）

VizieR 経由で **Pan-STARRS DR2 (II/349)** を検索。  
主等級は **PS1 r バンド**。カラーインデックスとして g−r も取得する。  
V バンドは存在しないため、V 等級が必要な場合は Gaia または変換式を別途使うこと。  
デカリネーション ≳ −30° の天域のみカバー。

### Gaia

VizieR 経由で **Gaia DR3 (I/355/gaiadr3)** を検索。  
主等級は **Gaia G バンド**。カラーインデックスとして BP−RP も取得する。  
全天対応。南天の小惑星にも使える。

### 2MASS

VizieR 経由で **2MASS PSC (II/246/out)** を検索。  
デフォルト主等級は **J バンド**。バンド選択で H / Ks に切り替えられる。  
カラーインデックスは J−Ks。近赤外観測の参照星確認に使う。

### SIMBAD

> **警告: SIMBAD は測光較正用カタログではありません。**  
> 参照星数の粗い確認や天体同定には使えるが、実際の測光較正には  
> Pan-STARRS / Gaia / 2MASS などの測光カタログを使うこと。  
> SIMBAD の object type は不完全であり、銀河や変光星が混入することがある。

---

## 出力 CSV 列

| 列名 | 説明 |
|------|------|
| `time` | 時刻ラベル（固定モードでは空欄） |
| `catalog` | 使用カタログ名 |
| `source_id` | カタログ ID |
| `ra_deg` | 赤経 (deg, ICRS) |
| `dec_deg` | 赤緯 (deg, ICRS) |
| `x_arcmin` | PA 回転後の幅方向オフセット (arcmin) |
| `y_arcmin` | PA 回転後の高さ方向オフセット (arcmin) |
| `separation_from_center_arcsec` | 中心からの角距離 (arcsec) |
| `mag` | 主バンド等級（Pan-STARRS: r、Gaia: G、2MASS: J/H/Ks） |
| `mag_err` | 等級誤差 |
| `color` | カラーインデックス（Pan-STARRS: g−r、Gaia: BP−RP、2MASS: J−Ks） |
| `object_type` | 天体タイプ（カタログ依存） |
| `usable` | `True` = 使用可、`False` = 除外 |
| `reject_reason` | 除外理由 |

---

## 判定基準

| 判定 | 使用可能星数 |
|------|-------------|
| **GOOD** | ≥ 30 |
| **OK** | 10–29 |
| **MARGINAL** | 5–9 |
| **POOR** | 1–4 |
| **BAD** | 0 |

---

## 注意事項

- **実際の画像では事前見積もりより使える星数が減ることがある。** 飽和、bad pixel、星のブレンド、small trail、月明かり、雲、airmass、大気吸収などが影響する。
- **non-sidereal tracking では背景星が流れる可能性がある。** 移動速度と露光時間から star trailing が無視できるか別途確認すること。
- **測光精度を主張するには、観測後に実際のフレームで参照星残差を評価する必要がある。** このツールはあくまで事前確認用である。
- **Pan-STARRS は赤緯 ≳ −30° のみ**カバーする。南天の小惑星には Gaia または 2MASS を使うこと。
- VizieR / SIMBAD の API が一時的に落ちている場合はエラーメッセージを表示して判定は `BAD` とする。

---

## 依存ライブラリ

- [astropy](https://www.astropy.org/)
- [astroquery](https://astroquery.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
