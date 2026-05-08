import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
import engine

def main():
    print("Khởi tạo vật liệu Sắt (Fe) cấu trúc BCC...")
    element = "Fe"
    phase = "bcc"
    atoms = bulk(element, phase, a=2.87)

    print("\n--- 1. Đang chạy tính toán Equation of State (Bulk Modulus) ---")
    volumes, energies, eos, v0, B_GPa = engine.compute_eos(atoms.copy(), pct_range=0.1, npts=10)
    print(f"-> Thể tích tối ưu: {v0:.4f} Å³")
    print(f"-> Bulk Modulus (B): {B_GPa:.2f} GPa")
    
    # Vẽ biểu đồ EOS
    plt.figure()
    plt.plot(volumes, energies, 'o', label='Dữ liệu')
    v_fit = np.linspace(min(volumes), max(volumes), 100)
    try:
        e_fit = eos.eos(v_fit, eos.eos_string, eos.v0, eos.e0, eos.B, eos.B1)
        plt.plot(v_fit, e_fit, '-', label='Birch-Murnaghan Fit')
    except Exception:
        pass
    plt.xlabel("Volume (Å³)")
    plt.ylabel("Total Energy (eV)")
    plt.title(f"Equation of State — {element} ({phase})")
    plt.legend()
    plt.show()

    print("\n--- 2. Đang chạy thử nghiệm Kéo (Tensile Test) ---")
    # Lấy một hệ nguyên tử mới để test kéo giãn
    atoms_tensile = bulk(element, phase, a=2.87)
    strains, stresses = engine.run_tensile_test(atoms_tensile, strain_max=0.10, steps=10)
    
    if len(strains) > 2:
        slope, _ = np.polyfit(strains[:3], stresses[:3], 1)
        print(f"-> Ước tính Young's Modulus: {slope:.2f} GPa")
        
    plt.figure()
    plt.plot(strains * 100, stresses, 'o-r')
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title(f"Tensile Test — {element} ({phase})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print("\n--- 3. Đang chạy Động lực học phân tử (Diffusion/MSD) ---")
    atoms_md = bulk(element, phase, a=2.87)
    T = 1000 # Nhiệt độ cực cao để thấy sự dịch chuyển
    times, msd_vals, D = engine.compute_msd_diffusion(atoms_md, T=T, steps=500, num_gpus=0)
    print(f"-> Hệ số khuếch tán ước tính (D) tại {T}K: {D:.4e} Å²/ps")
    
    plt.figure()
    plt.plot(times, msd_vals, 'b-', linewidth=2)
    plt.xlabel("Time (ps)")
    plt.ylabel("Mean Square Displacement (Å²)")
    plt.title(f"Diffusion (MSD) tại {T}K — {element} ({phase})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print("\nHoàn tất quá trình Test!")

if __name__ == "__main__":
    main()
