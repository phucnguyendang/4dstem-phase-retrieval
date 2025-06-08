# 4D-STEM Phase Recovery Deep Learning Framework

## Mô tả dự án

Dự án này triển khai mô hình deep learning để khôi phục phase từ các diffraction patterns (DP) thu được thông qua quá trình mô phỏng 4D-STEM. Hệ thống bao gồm hai giai đoạn chính:

1. **Patch Recovery**: Khôi phục phase patches từ 14×14 diffraction patterns
2. **Phase Stitching**: Ghép các phase patches thành ảnh phase đầy đủ 256×256

## Kiến trúc mô hình

### Step 1: PatchRecoveryNet
- **Input**: 14×14 DP partitions + coordinates  
- **Output**: 76×76 phase patches
- **Architecture**: ResNet34 backbone + ViT encoder + CNN decoder
- **Features**: Sử dụng pretrained ResNet34 và ViT để convergence nhanh

### Step 2: PhaseStitchingNet
- **Input**: Multiple 76×76 phase patches + coordinates
- **Output**: Full 256×256 phase images  
- **Architecture**: ViT-based encoder + CNN decoder
- **Features**: Hiểu spatial relationships giữa các patches

### Step 3: End2EndModel
- **Architecture**: Kết hợp PatchRecoveryNet + PhaseStitchingNet
- **Training Strategy**: Progressive training từ components đến end-to-end
- **Features**: Flexible freezing và fine-tuning strategies

## Cấu trúc dữ liệu

Dữ liệu được tổ chức trong 5 file HDF5:
- `simulation_data1.h5` - `simulation_data5.h5`
- Mỗi file chứa 700 samples
- Mỗi sample bao gồm:
  - 2500 diffraction patterns (50×50 grid, sampling step = 2Å)
  - 1 ảnh phase tương ứng (256×256, 100Å)

## Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd AI4Physics/Model

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cài đặt CUDA (tùy chọn, để tăng tốc training)

```bash
# Cho CUDA 11.x
pip install cupy-cuda11x

# Cho CUDA 12.x  
pip install cupy-cuda12x
```

### 3. Kiểm tra cài đặt

```bash
# Chạy test suite
jupyter notebook test.ipynb
```

## Sử dụng

### Training Strategy

#### Bước 1: Train PatchRecoveryNet riêng

```bash
python patchRecovery.py \
    --data_dir "path/to/simulation_data" \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir "./checkpoints/patch_recovery"
```

**Tham số quan trọng:**
- `--data_dir`: Thư mục chứa các file simulation_data*.h5
- `--batch_size`: Kích thước batch (khuyến nghị: 8-32)
- `--epochs`: Số epochs training
- `--lr`: Learning rate (khuyến nghị: 1e-4 đến 1e-5)

#### Bước 2: Train PhaseStitchingNet riêng

```bash
python PhaseStictchinng.py \
    --data_dir "path/to/simulation_data" \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir "./checkpoints/phase_stitching"
```

**Lưu ý:** Batch size nhỏ hơn do memory requirement cao hơn (25 patches per sample)

#### Bước 3: Train End-to-End

```bash
python End2End.py \
    --data_dir "path/to/simulation_data" \
    --patch_recovery_checkpoint "./checkpoints/patch_recovery/patch_recovery_best.pth" \
    --phase_stitching_checkpoint "./checkpoints/phase_stitching/phase_stitching_best.pth" \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-5 \
    --freeze_initial_epochs 10 \
    --save_dir "./checkpoints/end2end"
```

**Tham số đặc biệt:**
- `--freeze_initial_epochs`: Số epochs freeze PatchRecoveryNet (fine-tuning strategy)
- `--patch_weight`: Trọng số cho patch-level loss (default: 1.0)
- `--full_weight`: Trọng số cho full-phase loss (default: 1.0)

### Evaluation

```bash
python End2End.py evaluate \
    --data_dir "path/to/simulation_data" \
    --model_checkpoint "./checkpoints/end2end/end2end_best.pth" \
    --batch_size 4 \
    --output_dir "./results"
```

### Data Generation (nếu cần)

```bash
python gendata.py \
    --start_id 0 \
    --end_id 699 \
    --cell_size 100.0 \
    --cell_depth 15.0 \
    --probe_energy 300000 \
    --wave_function_size 256 \
    --semiangle_cutoff 30.0 \
    --defocus 0.0 \
    --sampling_step 2.0 \
    --num_phonon_configs 10 \
    --spherical_aberration 1000000 \
    --input_file "materials.h5" \
    --output_file "simulation_data1.h5"
```

## Cấu trúc files

```
AI4Physics/Model/
├── patchRecovery.py      # PatchRecoveryNet implementation
├── PhaseStictchinng.py   # PhaseStitchingNet implementation  
├── End2End.py            # End-to-end model combination
├── gendata.py            # 4D-STEM simulation data generation
├── test.ipynb            # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── checkpoints/         # Model checkpoints (created during training)
    ├── patch_recovery/
    ├── phase_stitching/
    └── end2end/
```

## Performance & Hardware Requirements

### Minimum Requirements
- **RAM**: 16GB+ (32GB khuyến nghị)
- **GPU**: NVIDIA GPU với 8GB+ VRAM
- **Storage**: 100GB+ cho dữ liệu và checkpoints
- **Python**: 3.8+

### Performance Benchmarks
- **PatchRecoveryNet**: ~50ms/batch (batch_size=16)
- **PhaseStitchingNet**: ~100ms/batch (batch_size=8)  
- **End2EndModel**: ~200ms/batch (batch_size=4)

### Memory Usage
- **Training**: 8-12GB GPU memory (tùy batch size)
- **Inference**: 4-6GB GPU memory
- **Data loading**: 4-8GB RAM per worker

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Giảm batch size
--batch_size 4  # thay vì 8

# Hoặc sử dụng gradient accumulation
--gradient_accumulation_steps 2
```

#### 2. Slow Data Loading
```bash
# Tăng số workers (nhưng cẩn thận với RAM)
--num_workers 4

# Hoặc giảm nếu bị bottleneck
--num_workers 2
```

#### 3. Model Not Converging
```bash
# Thử learning rate nhỏ hơn
--lr 5e-5

# Hoặc sử dụng learning rate scheduling
# (đã được tích hợp sẵn)
```

#### 4. Import Errors
```bash
# Reinstall transformers nếu có lỗi ViT
pip uninstall transformers
pip install transformers>=4.20.0

# Cài đặt abtem nếu có lỗi simulation
pip install abtem
```

## Monitoring Training

### TensorBoard (tùy chọn)
```bash
# Thêm vào training scripts nếu muốn
pip install tensorboard
# Code sẽ cần được modify để log metrics
```

### Training Progress
- Loss được in ra mỗi 50 batches (PatchRecovery) hoặc 20 batches (End2End)
- Validation loss được tính sau mỗi epoch
- Best models được save tự động
- Checkpoints được save mỗi 10 epochs

## Advanced Usage

### Custom Loss Functions
Modify loss functions trong End2End.py:
```python
# Thay đổi trọng số loss
criterion = End2EndLoss(patch_weight=2.0, full_weight=1.0)

# Hoặc thêm regularization
```

### Model Architecture Modifications
- Thay đổi embed_dim trong PhaseStitchingNet
- Modify ResNet backbone trong PatchRecoveryNet
- Adjust ViT layers và attention heads

### Data Augmentation
Thêm augmentation vào Dataset classes:
```python
# Random rotation, flip, noise addition
# Đặc biệt hữu ích cho robustness
```

## Citation & References

Nếu sử dụng code này trong nghiên cứu, vui lòng cite:

```bibtex
@misc{4dstem_phase_recovery,
  title={Deep Learning Framework for 4D-STEM Phase Recovery},
  author={Your Name},
  year={2025},
  institution={Hanoi University of Science and Technology}
}
```

## License

Dự án này được phát hành dưới giấy phép MIT. Xem file LICENSE để biết thêm chi tiết.

## Contact & Support

- **Issues**: Tạo issue trên GitHub repository
- **Email**: your.email@hust.edu.vn
- **Documentation**: Xem test.ipynb để hiểu rõ hơn về API

## Changelog

### v1.0.0 (2025-06-07)
- Initial release
- PatchRecoveryNet implementation
- PhaseStitchingNet implementation  
- End-to-end training pipeline
- Comprehensive test suite
- Documentation và setup scripts
