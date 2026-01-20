import numpy as np
from params import SourceParams
from models import IREModel, generate_grid

def main():
    # 1. 定义单模型参数 (Base Parameters)
    params_dict = dict(
        outputfilename="example.fits",
        
        # 观测参数
        fldres_as=0.05,          # 空间分辨率 (arcsec)
        velres_kmps=0.686,       # 速度分辨率 (km/s)
        distance_pc=5100.0,      # 距离 (pc)
        beam_maj_as=0.37451,     # 波束长轴 (arcsec)
        beam_min_as=0.29969,     # 波束短轴 (arcsec)
        beam_pa_deg=54.3281,     # 波束位置角 (deg)
        linewidth_kmps=1.0,      # 内禀线宽 (km/s)
        
        # 物理参数 - 中心天体
        mass_msun=450.0,         # 原恒星质量 (Msun)
        rcb_au=2000.0,           # 离心势垒半径 (au)
        
        # 几何参数
        inc_deg=-45.0,           # 倾角 (deg)
        pa_deg=160.0,            # 位置角 (deg)
        rot_sign=1.0,            # 旋转方向 (1 or -1)
        rout_au=15000.0,         # 外半径 (au)
        rin_au=1000.0,           # 内半径 (au)
        
        # IRE (内落包层) 参数
        height_ire_au=0.0,       # 标高
        flare_ire_deg=30.0,      # 张角
        density_profile_ire=-1.5, # 密度幂律指数
        temp_profile_ire=0.0,    # 温度幂律指数
        
        # Keplerian Disk (开普勒盘) 参数
        height_kep_au=0.0,
        flare_kep_deg=30.0,
        density_profile_kep=-1.5,
        temp_profile_kep=0.0,
        
        # 基准值 (在 rCB 处)
        dens_cb=1.0e-2,          # 密度 (cm^-3)
        temp_cb=10.0,            # 温度 (K)
        
        # 谱线信息
        name_line="CH3OH",
        restfreq_ghz=1.0,   # 目标线的静止频率 in GHz
        name_object="test",
        radesys="ICRS",
        center_ra_str="16h09m52.5704s",
        center_dec_str="-51d54m54.3644s",
        vsys_kmps=0.0
    )

    # 2. 单个模型生成示例 (Single Model Generation)
    print("\n--- Starting Single Model Demo ---")
    
    # 从字典创建参数对象
    # 注意：params_dict 中的 outputfilename 会被 SourceParams 使用
    params = SourceParams(**params_dict)
    
    print(f"Parsed Dec String: {params.center_dec_str}")
    print(f"Calculated Dec (deg): {params.cent_dec_deg}")
    
    # 初始化模型
    print("Initializing Model...")
    model = IREModel(params, npix=128, nvel=64)
    
    # 生成数据立方体 (Cube)
    model.make_cube(output_filename="xfw.fits")
    
    # 生成 PV 图
    model.make_pv(
        pv_pa_deg=338.0,      # 切片方向
        pv_off_ra=0.0,        # RA 偏移 (au)
        pv_off_dec=0.0,       # Dec 偏移 (au)
        output_filename="example_PV.fits",
        plot=True
    )
    
    # 生成 Moment 图
    model.make_moments(output_prefix="example", moment_type='all', threshold=0.01, plot=True)

    # 3. 演示网格生成 (Grid Generation)
    print("\n--- Starting Grid Generation Demo ---")
    
    # 设定变量范围
    # 改变 mass_msun 和 rcb_au
    grid_params = params_dict.copy()
    grid_params['mass_msun'] = np.array([400.0, 500.0])   # 两个质量点
    grid_params['rcb_au'] = np.array([1500.0, 2500.0])    # 两个半径点
    
    # 调用网格生成函数
    # 结果将保存在 grid_output 文件夹下
    generate_grid(
        grid_params, 
        output_dir="grid_demo_output", 
        prefix="example_grid",
        npix=128, 
        nvel=64
    )
    

if __name__ == "__main__":
    main()
