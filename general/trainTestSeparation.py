'''
 * Copyright (C) 2018  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''


def get_train_test_recordings_joint():
    """train_nacta_2017, train_nacta for syllable and phoneme joint estimation and HSMM alignment acoustic model"""

    # training recordings names in nacta 2017 dataset, for both HSMM and joint models
    train_nacta_2017 = [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],  # yes, pro in the artist album test
                      ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta'],  # pro in the artist album test
                      ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta']]  # yes # pro

    # training recording names in nacta dataset, for both HSMM and joint model
    train_nacta = [['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                   ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                   ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                   ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                   ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                   ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'],
                   ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                   ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                   ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'],  # amateur
                   ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'],  # amateur
                   ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'],  # amateur
                   ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],  # amateur
                   ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                   ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                   ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm'],
                   ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'], # amateur
                   ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],  # amateur
                   ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'],  # amateur
                   ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],  # amateur
                   ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm'],  # amateur
                   ['danAll', 'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon'],  # no amateur, in the artist album test
                   ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'],  # yes amateur, in the artist album test
                   ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'],  # yes amateur in the artist album test
                   ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf']]  # amateur in the artist album test

    # training recording names only for training hSMM acoustic model
    train_sepa = [['danAll', 'shiwenhui_tingxiongyan'],
                  ['danAll', 'xixiangji_biyuntian'],
                  ['danAll', 'xixiangji_diyilai'],
                  ['danAll', 'xixiangji_luanchouduo'],
                  ['danAll', 'xixiangji_manmufeng'],
                  ['danAll','xixiangji_xianzhishuo'],
                  ['danAll', 'xixiangji_zhenmeijiu'],
                  ['danAll', 'yutangchun_yutangchun'],
                  ['danAll', 'zhuangyuanmei_daocishi'],
                  ['danAll', 'zhuangyuanmei_fudingkui'],
                  ['danAll', 'zhuangyuanmei_tianbofu'],
                  ['danAll', 'zhuangyuanmei_zhenzhushan'],
                  ['danAll', 'zhuangyuanmei_zinari'],
                  ['danAll', 'wangjiangting_zhijianta'],
                  ['danAll', 'wangjiangting_dushoukong']
                 ]

    # train recording names in primary school dataset, for both HSMM and joint models
    train_primary_school = [['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'teacher'],
                            ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'teacher'],
                            ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'teacher'],
                            ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'teacher'],
                            ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'teacher'],
                            ['20171214SongRuoXuan/danbz-Kan_dai_wang-Ba_wang_bie_ji-nanluo', 'teacher'],
                            ['20171214SongRuoXuan/daspd-Hai_dao_bing-Gui_fei_zui_jiu-nanluo', 'teacher'],
                            ['20171214SongRuoXuan/daxp-Quan_jun_wang-Ba_wang_bie_ji-nanluo', 'teacher'],
                            ['20171215SongRuoXuan/daxp-Jiao_zhang_sheng-Xi_shi-qianmen', 'teacher'],
                            ['2017121215SongRuoXuan/daxp-Mu_qin_bu_ke-Feng_huan_chao-yucai_qianmen', 'teacher'],

                            ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'teacher'],
                            ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'teacher'],
                            ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'teacher'],
                            ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'teacher']]

    # validation recording names in primary school dataset to tune the parameters, fixed for the submission interspeech
    val_primary_school = [['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student01'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02_first_half'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student03'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student04'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student05'],
                          ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student06'],

                          ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student01'],
                          ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student02'],
                          ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student03'],
                          ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student04'],

                          ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'student_01'],
                          ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'student_02'],

                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_01'],
                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_02'],
                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_03'],
                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_04'],
                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_05']]

    # test recording names in primary school dataset, fixed for the submission interspeech
    test_primary_school = [['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student01'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student02'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student03'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student04'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student05'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student06'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student07'],
                           ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student08'],

                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student01'],
                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student02'],
                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student03'],
                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student04'],
                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student05'],
                           ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student06'],

                           ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_01'],
                           ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_02'],
                           ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_1'],
                           ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_2'],
                           ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_04_mentougou'],

                           ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'student_01'],
                           ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'student_02']]

    return val_primary_school, test_primary_school, train_nacta_2017, train_nacta, train_primary_school, train_sepa


def get_train_test_recordings_joint_subset():
    """train_nacta_2017, train_nacta for syllable and phoneme joint estimation and HSMM alignment acoustic model,
    a subset, for another forgot purpose."""

    train_nacta_2017 = [['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta']]  # yes pro

    train_nacta = [['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                   ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                   ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                   ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                   ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                   ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'],
                   ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                   ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                   ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'],  # amateur
                   ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'],  # amateur
                   ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'],  # amateur
                   ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],  # amateur
                   ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                   ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                   ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm'],
                   ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'],  # amateur
                   ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],   # amateur
                   ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'],  # amateur
                   ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],  # amateur
                   ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm'],  # amateur
                   ]

    return train_nacta_2017, train_nacta