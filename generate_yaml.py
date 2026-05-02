import json

JSON_FILE = 'data/raw_annotations/train_annotations.json'
YAML_FILE = 'data/vista.yaml'

def make_yaml():
    with open(JSON_FILE) as f:
        data = json.load(f)
        
    cats = data.get('categories', [])
    
    # Sort by 'id' to ensure order matches the converter
    cats.sort(key=lambda x: x.get('ind', x['id']))
    
    print(f"Generating YAML for {len(cats)} classes...")
    
    with open(YAML_FILE, 'w') as f:
        f.write("path: ./data\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("test:  images/test\n\n")
        f.write("names:\n")
        
        for c in cats:
            # Write "  0: name" using either ind or id
            idx = c.get('ind', c['id'])
            f.write(f"  {idx}: \"{c['name']}\"\n")

    print(f"✅ Saved config to {YAML_FILE}")

if __name__ == '__main__':
    make_yaml()
