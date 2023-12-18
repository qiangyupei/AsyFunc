<!-- wordir: /path/to/coordinator -->
sudo docker buildx build --platform linux/amd64,linux/arm64 -t &lt;image_repository_url&gt;/&lt;Author&gt;/coordinator:&lt;version&gt; --push .