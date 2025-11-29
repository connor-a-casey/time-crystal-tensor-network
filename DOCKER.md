# Docker Quick Reference

This file provides quick reference commands for using Docker with this project.

## Building the Image

```bash
docker build -t time-crystal-tensor-network .
```

## Running Commands

### Run Tests
```bash
docker run --rm time-crystal-tensor-network python tests/run_tests.py
```

### Generate All Figures
```bash
docker run --rm -v $(pwd)/figures:/app/figures time-crystal-tensor-network python main.py
```

### Interactive Shell
```bash
docker run -it --rm -v $(pwd)/figures:/app/figures time-crystal-tensor-network bash
```

## Using Docker Compose

### Build and Start
```bash
docker-compose up -d
```

### Run Tests
```bash
docker-compose run --rm time-crystal python tests/run_tests.py
```

### Generate Figures
```bash
docker-compose run --rm time-crystal python main.py
```

### Interactive Shell
```bash
docker-compose run --rm time-crystal bash
```

### Stop and Cleanup
```bash
docker-compose down
```

## Volume Mounts

The `-v $(pwd)/figures:/app/figures` flag mounts your local `figures/` directory into the container, allowing generated figures to persist on your host machine.

## Troubleshooting

- **Permission errors**: Ensure Docker has proper permissions (may need `sudo` on Linux)
- **Build fails**: Check internet connection and Docker daemon status
- **Out of space**: Clean up unused images: `docker system prune -a`

