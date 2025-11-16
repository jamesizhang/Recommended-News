import unittest
import pandas as pd
import numpy as np


class TestIsCatHab(unittest.TestCase):
    def setUp(self):
        # 构造最小可复现数据
        self.trn_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'click_article_id': [101, 102, 103, 104],
            'category_id': [10, 20, 30, 40],
            # cate_list 形态覆盖：list / set / 非法类型
            'cate_list': [
                [10, 11, 12],          # 命中（10 in list）
                set([21, 22, 23]),     # 未命中（20 not in set）
                (30, 31),              # 命中（30 in tuple）
                np.nan                 # 非法/缺失，按未命中处理
            ]
        })

    def _calc_is_cat_hab_local(self, df: pd.DataFrame) -> pd.Series:
        result = []
        for _, row in df[['category_id', 'cate_list']].iterrows():
            cate_list = row['cate_list']
            if isinstance(cate_list, (list, set, tuple)):
                result.append(1 if row['category_id'] in set(cate_list) else 0)
            else:
                result.append(0)
        return pd.Series(result, index=df.index)

    def test_is_cat_hab_values(self):
        # 计算期望值
        expected = self._calc_is_cat_hab_local(self.trn_df)

        # 模拟生产逻辑：赋值到单列不应报错，且结果一致
        df = self.trn_df.copy()
        df['is_cat_hab'] = self._calc_is_cat_hab_local(df)

        self.assertIn('is_cat_hab', df.columns)
        self.assertEqual(df['is_cat_hab'].dtype, expected.dtype)
        self.assertTrue((df['is_cat_hab'].values == expected.values).all())
        # 逐行校验
        self.assertEqual(df.loc[0, 'is_cat_hab'], 1)  # 命中
        self.assertEqual(df.loc[1, 'is_cat_hab'], 0)  # 未命中
        self.assertEqual(df.loc[2, 'is_cat_hab'], 1)  # 命中
        self.assertEqual(df.loc[3, 'is_cat_hab'], 0)  # 缺失/非法按未命中


if __name__ == '__main__':
    unittest.main(verbosity=2)
