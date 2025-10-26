import cv2
import numpy as np
from sklearn.cluster import KMeans, MeanShift
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
import os
import argparse
import sys

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def count_unique_colors(image):
    """计算图像中唯一颜色的数量"""
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)


def compress_colors_kmeans(image, n_colors):
    """使用K-Means算法进行颜色压缩

    Args:
        image: RGB图像数组
        n_colors: 目标颜色数量

    Returns:
        压缩后的图像和耗时
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    start_time = time.time()
    try:
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        compressed_pixels = kmeans.cluster_centers_[labels]
        elapsed_time = time.time() - start_time

        compressed_image = compressed_pixels.reshape(h, w, c).astype(np.uint8)
        return compressed_image, elapsed_time
    except Exception as e:
        print(f"K-Means压缩失败 (k={n_colors}): {e}")
        return None, 0


def compress_colors_meanshift(image, bandwidth=None, sample_size=10000):
    """使用Mean Shift算法进行颜色压缩

    Args:
        image: RGB图像数组
        bandwidth: 带宽参数,None则自动估计
        sample_size: 采样像素数量,用于加速处理

    Returns:
        压缩后的图像、耗时和聚类中心数
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    # 为了加快处理速度,采样部分像素
    actual_sample_size = min(sample_size, len(pixels))
    sample_indices = np.random.choice(len(pixels), actual_sample_size, replace=False)
    sample_pixels = pixels[sample_indices]

    start_time = time.time()
    try:
        if bandwidth is None:
            from sklearn.cluster import estimate_bandwidth

            bandwidth = estimate_bandwidth(sample_pixels, quantile=0.1, n_samples=500)
            print(f"  自动估计带宽: {bandwidth:.2f}")

        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(sample_pixels)

        # 预测所有像素
        labels = meanshift.predict(pixels)
        compressed_pixels = meanshift.cluster_centers_[labels]
        elapsed_time = time.time() - start_time

        compressed_image = compressed_pixels.reshape(h, w, c).astype(np.uint8)
        n_colors = len(meanshift.cluster_centers_)

        return compressed_image, elapsed_time, n_colors
    except Exception as e:
        print(f"Mean Shift压缩失败 (bandwidth={bandwidth}): {e}")
        return None, 0, 0


def load_and_preprocess_image(image_path, max_size=1200):
    """加载并预处理图像

    Args:
        image_path: 图像路径
        max_size: 最大尺寸,超过则缩放

    Returns:
        RGB格式的图像数组,失败返回None
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 '{image_path}'")
        return None

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 '{image_path}'")
            return None

        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 如果图像过大,进行缩放以提高处理速度
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"图像已缩放: {w}x{h} -> {new_w}x{new_h}")

        return image
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None


def save_compressed_images(results, prefix, output_dir="compressed_images"):
    """保存压缩后的图像

    Args:
        results: 包含压缩结果的字典
        prefix: 文件名前缀
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, result in results.items():
        if result["image"] is not None:
            filename = f"{prefix}_k{key}.png"
            filepath = os.path.join(output_dir, filename)
            # 转换回BGR保存
            image_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, image_bgr)
    print(f"压缩图像已保存到 '{output_dir}/' 目录")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="图像颜色压缩 - K-Means vs Mean Shift")
    parser.add_argument(
        "--image", type=str, default="your_photo.jpg", help="输入图像路径"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[20, 15, 10, 7, 5, 3, 2],
        help="K-Means的k值列表",
    )
    parser.add_argument(
        "--bandwidths",
        type=int,
        nargs="+",
        default=[50, 40, 35, 30, 25, 20, 15],
        help="Mean Shift的带宽值列表",
    )
    parser.add_argument(
        "--max_size", type=int, default=1200, help="图像最大尺寸,超过则缩放"
    )
    parser.add_argument(
        "--save_images", action="store_true", help="是否保存压缩后的图像"
    )
    parser.add_argument("--no_show", action="store_true", help="不显示图像窗口")

    args = parser.parse_args()

    # 读取图片
    print(f"正在加载图像: {args.image}")
    image = load_and_preprocess_image(args.image, max_size=args.max_size)
    if image is None:
        return 1

    # (1) 计算原始颜色数量
    print("\n正在分析原始图像...")
    original_colors = count_unique_colors(image)
    print(f"原始照片尺寸: {image.shape[1]}x{image.shape[0]}")
    print(f"原始照片中的颜色数量: {original_colors}")

    # (2) 使用不同的k值进行压缩
    k_values = args.k_values

    # K-Means 结果
    print("\n" + "=" * 50)
    print("=== K-Means 聚类算法 ===")
    print("=" * 50)
    kmeans_results = {}
    for k in k_values:
        print(f"处理 K={k}...", end=" ", flush=True)
        compressed, elapsed = compress_colors_kmeans(image, k)
        if compressed is not None:
            actual_colors = count_unique_colors(compressed)
            kmeans_results[k] = {
                "image": compressed,
                "time": elapsed,
                "colors": actual_colors,
            }
            print(f"✓ 耗时 {elapsed:.2f}秒, 实际颜色数 {actual_colors}")
        else:
            print("✗ 失败")

    # Mean Shift 结果（使用不同的bandwidth来近似不同的k值）
    print("\n" + "=" * 50)
    print("=== Mean Shift 聚类算法 ===")
    print("=" * 50)
    bandwidths = args.bandwidths
    meanshift_results = {}
    for bw in bandwidths:
        print(f"处理 Bandwidth={bw}...")
        compressed, elapsed, n_colors = compress_colors_meanshift(image, bandwidth=bw)
        if compressed is not None:
            actual_colors = count_unique_colors(compressed)
            meanshift_results[bw] = {
                "image": compressed,
                "time": elapsed,
                "colors": actual_colors,
                "clusters": n_colors,
            }
            print(
                f"  ✓ 耗时 {elapsed:.2f}秒, 聚类中心数 {n_colors}, 实际颜色数 {actual_colors}"
            )
        else:
            print("  ✗ 失败")

    if not kmeans_results and not meanshift_results:
        print("\n错误: 所有压缩操作均失败")
        return 1

    # 保存压缩图像（可选）
    if args.save_images:
        print("\n正在保存压缩图像...")
        if kmeans_results:
            save_compressed_images(kmeans_results, "kmeans")
        if meanshift_results:
            save_compressed_images(meanshift_results, "meanshift")

    # (3) 可视化对比
    print("\n正在生成可视化对比图...")
    # K-Means 结果展示
    if kmeans_results:
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        fig1.suptitle("K-Means Clustering Color Compression", fontsize=16)
        axes1[0, 0].imshow(image)
        axes1[0, 0].set_title(f"Original\n({original_colors} colors)")
        axes1[0, 0].axis("off")

        for idx, k in enumerate(sorted(kmeans_results.keys())[:7], 1):
            row = idx // 4
            col = idx % 4
            axes1[row, col].imshow(kmeans_results[k]["image"])
            axes1[row, col].set_title(
                f'K={k}\n({kmeans_results[k]["colors"]} colors)\nTime: {kmeans_results[k]["time"]:.2f}s'
            )
            axes1[row, col].axis("off")

        plt.tight_layout()
        plt.savefig("kmeans_comparison.png", dpi=150, bbox_inches="tight")
        print("✓ K-Means 对比图已保存为 kmeans_comparison.png")

    # Mean Shift 结果展示
    if meanshift_results:
        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle("Mean Shift Clustering Color Compression", fontsize=16)
        axes2[0, 0].imshow(image)
        axes2[0, 0].set_title(f"Original\n({original_colors} colors)")
        axes2[0, 0].axis("off")

        for idx, bw in enumerate(sorted(meanshift_results.keys())[:7], 1):
            row = idx // 4
            col = idx % 4
            axes2[row, col].imshow(meanshift_results[bw]["image"])
            axes2[row, col].set_title(
                f'BW={bw}\n({meanshift_results[bw]["colors"]} colors)\nTime: {meanshift_results[bw]["time"]:.2f}s'
            )
            axes2[row, col].axis("off")

        plt.tight_layout()
        plt.savefig("meanshift_comparison.png", dpi=150, bbox_inches="tight")
        print("✓ Mean Shift 对比图已保存为 meanshift_comparison.png")

    # 性能对比分析
    print("\n" + "=" * 50)
    print("=== 算法性能对比分析 ===")
    print("=" * 50)

    if kmeans_results and meanshift_results:
        print("\n1. 运行时间对比:")
        kmeans_avg = np.mean([r["time"] for r in kmeans_results.values()])
        meanshift_avg = np.mean([r["time"] for r in meanshift_results.values()])
        print(f"K-Means 平均耗时: {kmeans_avg:.2f} 秒")
        print(f"Mean Shift 平均耗时: {meanshift_avg:.2f} 秒")
        print(f"速度对比: K-Means 比 Mean Shift 快 {meanshift_avg/kmeans_avg:.1f}x")

    print("\n2. 算法特点:")
    print("K-Means:")
    print("  - 优点: 速度快，可精确控制颜色数量")
    print("  - 缺点: 需要预先指定聚类数，对初始值敏感")

    print("\nMean Shift:")
    print("  - 优点: 无需预先指定聚类数，能自动发现聚类中心")
    print("  - 缺点: 计算速度较慢，带宽参数需要调整")

    print("\n3. 压缩效果对比:")
    print("K-Means 提供了更可控的颜色压缩效果，适合需要精确控制输出颜色数的场景")
    print("Mean Shift 能够根据图像特征自适应地确定颜色数量，保留更多细节")

    if not args.no_show:
        print("\n显示图像窗口（关闭窗口继续）...")
        plt.show()

    print("\n✓ 处理完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
