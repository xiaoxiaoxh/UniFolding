using ClothDynamics;
using DG.Tweening;
using RFUniverse;
using RFUniverse.Attributes;
using Robotflow.RFUniverse.SideChannels;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class Main : MonoBehaviour
{
    public CameraAttr[] cameras;
    public GPUClothDynamicsV2 clothDynamicsV2;
    public MeshFilter depthMesh;
    public float lowGravity = 2;
    public float midGravity = 6;
    public float highGravity = 15;
    private float particleRadius;

    GPUClothDynamicsV2 currentClothDynamicsV2 = null;
    ClothObjectGPU currentCloth = null;

    void Awake()
    {
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetRandomPosition", msg => GetRandomPosition());
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetInitParticles", msg => GetInitParticles(msg.ReadFloat32()));
        RFUniverse.Manager.AssetManager.Instance.AddListener("LoadMesh", msg => StartCoroutine(LoadMesh(msg.ReadString(), msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32())));
        RFUniverse.Manager.AssetManager.Instance.AddListener("AddAttach", msg => AddAttach(new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32())));
        RFUniverse.Manager.AssetManager.Instance.AddListener("RemoveAllAttach", msg => RemoveAllAttach());
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetDepth", msg => GetDepth(msg.ReadFloatList().ToList()));
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetParticles", msg => GetParticles());
        RFUniverse.Manager.AssetManager.Instance.AddListener("SetGravity", msg => SetGravity(new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32())));
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetGraspPosition", msg => GetGraspPosition());
        RFUniverse.Manager.AssetManager.Instance.AddListener("GetParticleRadius", msg => GetParticleRadius());
        RFUniverse.Manager.AssetManager.Instance.AddListener("DualMove", msg => DualMove(
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32()),
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32()),
                msg.ReadFloat32()
                ));
        RFUniverse.Manager.AssetManager.Instance.AddListener("Grab", msg => StartCoroutine(
            Grab(
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32()),
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32())
                )));
        RFUniverse.Manager.AssetManager.Instance.AddListener("Stretch", msg => StartCoroutine(
            Stretch(
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32()),
                new Vector3(msg.ReadFloat32(), msg.ReadFloat32(), msg.ReadFloat32()),
                msg.ReadFloat32(),
                msg.ReadFloat32(),
                msg.ReadFloat32(),
                msg.ReadFloat32()
                )));
        RFUniverse.Manager.AssetManager.Instance.AddListener("RandomFold", msg => StartCoroutine(RandomFold(msg.ReadBoolean())));
    }
    public Vector3 GetRandomPosition()
    {
        Debug.Log("GetRandomPosition");
        Mesh mesh = currentCloth.MyGetMesh();
        Vector3[] positions = mesh.vertices;
        Vector3 position = positions[UnityEngine.Random.Range(0, positions.Length)];
        RFUniverse.Manager.AssetManager.Instance.SendMessage("RandomPosition", position.x, position.y, position.z);
        return position;
    }

    public void GetParticleRadius()
    {
        Debug.Log("GetParticleRadius");
        RFUniverse.Manager.AssetManager.Instance.SendMessage("ParticleRadius", particleRadius);
    }

    public void SetGravity(Vector3 gravity)
    {
        Debug.Log("SetGravity");
        currentClothDynamicsV2._globalSimParams.gravity = gravity;
    }

    public IEnumerator LoadMesh(string path, float particleScale = 1f, float theta = 0, float offsetX = 0, float offsetZ = 0)
    {
        Debug.Log("LoadMesh");
        if (currentClothDynamicsV2 != null)
        {
            Destroy(currentClothDynamicsV2.gameObject);
        }
        currentClothDynamicsV2 = Instantiate(clothDynamicsV2);

        if (currentCloth != null)
        {
            Destroy(currentCloth.gameObject);
        }
        GameObject obj = UnityMeshImporter.MeshImporter.Load(path);
        Mesh mesh = obj.GetComponentInChildren<MeshFilter>().mesh;
        int rawMeshVertexCount = mesh.vertices.Length;
        MeshRenderer render = obj.GetComponentInChildren<MeshRenderer>();
        currentClothDynamicsV2._globalSimParams.particleDiameterScalar = GetMeshPointAverageDistance(mesh) * particleScale;
        obj.transform.eulerAngles = new Vector3(0, theta, 0);
        Vector3 centerOffset = new Vector3(render.bounds.center.x, 0, render.bounds.center.z);
        obj.transform.position = Vector3.up * 0.3f - centerOffset;
        obj.transform.position += new Vector3(offsetX, 0, offsetZ);
        render.material.SetColor("_Color", Color.white);
        render.material.SetFloat("_Metallic", 0);
        Destroy(render.GetComponent<Collider>());
        render.transform.parent = null;
        Destroy(obj);

        currentClothDynamicsV2._solver._runSim = false;
        currentCloth = render.gameObject.AddComponent<ClothObjectGPU>();
        // wati for cloth dynamics to re-generate mesh vertices
        yield return RFUniverseUtility.WaitFixedUpdateFrame(50);

        Vector3[] vector3s = GetParticles(false);
        while (vector3s.Length == rawMeshVertexCount)
        {
            // Sleep until particles are updated
            yield return RFUniverseUtility.WaitFixedUpdateFrame(50);
            vector3s = GetParticles(false);
        }

        List<float> raw_vertices_list = RFUniverseUtility.ListVector3ToListFloat(
            vector3s.Select((s) => render.transform.InverseTransformPoint(s)).ToList());
        RFUniverse.Manager.AssetManager.Instance.SendMessage("StaticMeshVertices", raw_vertices_list);
        currentClothDynamicsV2._solver._runSim = true;

        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * lowGravity;
        yield return new WaitForSeconds(3);
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * highGravity;
        RFUniverse.Manager.AssetManager.Instance.SendMessage("Done");
        GetFeaturePoints(theta);
    }

    Vector3[] sourcePoints;
    Tuple<int, Vector3> sourceRightCuff;
    Tuple<int, Vector3> sourceLeftCuff;
    Tuple<int, Vector3> sourceRightHem;
    Tuple<int, Vector3> sourceLeftHem;
    Tuple<int, Vector3> sourceRightShoulder;
    Tuple<int, Vector3> sourceLeftShoulder;
    Tuple<int, Vector3> sourceCenter;
    float averagerHeight;
    void GetFeaturePoints(float theta = 0)
    {
        sourcePoints = GetParticles(false);
        averagerHeight = sourcePoints.Select((s) => s.y).Average();
        var inverseRotationPoints = sourcePoints.Select((s, i) => new Tuple<int, Vector3>(i, Quaternion.Euler(0, -theta, 0) * s));
        float[] zs = inverseRotationPoints.Select(s => s.Item2.z).ToArray();
        float[] xs = inverseRotationPoints.Select(s => s.Item2.x).ToArray();
        float maxZ = zs.Max();
        float minZ = zs.Min();
        float maxX = xs.Max();
        float minX = xs.Min();
        var topHalf = inverseRotationPoints.Where(s => s.Item2.z > maxZ - 0.1f);
        var rightCuff = inverseRotationPoints.Where(s => s.Item2.x > maxX - 0.03f).Aggregate((a, b) => a.Item2.z > b.Item2.z ? a : b);
        var leftCuff = inverseRotationPoints.Where(s => s.Item2.x < minX + 0.03f).Aggregate((a, b) => a.Item2.z > b.Item2.z ? a : b);
        var bottomHalf = inverseRotationPoints.Where(s => s.Item2.z < minZ + 0.1f);
        var rightHem = bottomHalf.Aggregate((a, b) => a.Item2.x > b.Item2.x ? a : b);
        var leftHem = bottomHalf.Aggregate((a, b) => a.Item2.x < b.Item2.x ? a : b);
        var rightShoulder = inverseRotationPoints.Where(s => Mathf.Abs(s.Item2.x - rightHem.Item2.x) < 0.03f).Aggregate((a, b) => a.Item2.z > b.Item2.z ? a : b);
        var leftShoulder = inverseRotationPoints.Where(s => Mathf.Abs(s.Item2.x - leftHem.Item2.x) < 0.03f).Aggregate((a, b) => a.Item2.z > b.Item2.z ? a : b);
        Vector3[] vector3s = new Vector3[] { rightHem.Item2, leftHem.Item2, rightShoulder.Item2, leftShoulder.Item2 };
        Vector3 centerPosition = (rightHem.Item2 + leftHem.Item2 + rightShoulder.Item2 + leftShoulder.Item2) / 4;
        var center = inverseRotationPoints.Aggregate((a, b) => Vector3.Distance(a.Item2, centerPosition) < Vector3.Distance(b.Item2, centerPosition) ? a : b);

        sourceRightCuff = new Tuple<int, Vector3>(rightCuff.Item1, Quaternion.Euler(0, theta, 0) * rightCuff.Item2);
        sourceLeftCuff = new Tuple<int, Vector3>(leftCuff.Item1, Quaternion.Euler(0, theta, 0) * leftCuff.Item2);
        sourceRightHem = new Tuple<int, Vector3>(rightHem.Item1, Quaternion.Euler(0, theta, 0) * rightHem.Item2);
        sourceLeftHem = new Tuple<int, Vector3>(leftHem.Item1, Quaternion.Euler(0, theta, 0) * leftHem.Item2);
        sourceRightShoulder = new Tuple<int, Vector3>(rightShoulder.Item1, Quaternion.Euler(0, theta, 0) * rightShoulder.Item2);
        sourceLeftShoulder = new Tuple<int, Vector3>(leftShoulder.Item1, Quaternion.Euler(0, theta, 0) * leftShoulder.Item2);
        sourceCenter = new Tuple<int, Vector3>(center.Item1, Quaternion.Euler(0, theta, 0) * center.Item2);
    }
    List<Vector3> keyPoint = new List<Vector3>();
    List<float> best = new List<float>();
    public IEnumerator RandomFold(bool is_short_sleeve = true)
    {
        Debug.Log("RandomFold");
        bool frontOrBack = UnityEngine.Random.value > 0.5f;
        Vector3[] points = GetParticles(false);
        if (frontOrBack)
        {
            int corner = UnityEngine.Random.Range(0, 4);
            Vector3 randomCenter;
            switch (corner)
            {
                case 0:
                    do
                    {
                        randomCenter = RandomPointInQuad(points[sourceRightCuff.Item1], points[sourceRightHem.Item1], points[sourceLeftCuff.Item1], points[sourceLeftHem.Item1]);
                    }
                    while (Vector3.Distance(randomCenter, points[sourceRightCuff.Item1]) < 0.2f);
                    yield return StartCoroutine(Fold(points[sourceRightCuff.Item1], randomCenter));
                    points = GetParticles(false);
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        best.Add(1);
                        best.Add(1);
                    }
                    break;
                case 1:
                    do
                    {
                        randomCenter = RandomPointInQuad(points[sourceRightCuff.Item1], points[sourceRightHem.Item1], points[sourceLeftCuff.Item1], points[sourceLeftHem.Item1]);
                    }
                    while (Vector3.Distance(randomCenter, points[sourceLeftCuff.Item1]) < 0.2f);
                    yield return StartCoroutine(Fold(points[sourceLeftCuff.Item1], randomCenter));
                    points = GetParticles(false);
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        best.Add(1);
                        best.Add(1);
                    }

                    break;
                case 2:
                    do
                    {
                        randomCenter = RandomPointInQuad(points[sourceRightCuff.Item1], points[sourceRightHem.Item1], points[sourceLeftCuff.Item1], points[sourceLeftHem.Item1]);
                    }
                    while (Vector3.Distance(randomCenter, points[sourceRightHem.Item1]) < 0.2f);
                    yield return StartCoroutine(Fold(points[sourceRightHem.Item1], randomCenter));
                    points = GetParticles(false);
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        best.Add(1);
                        best.Add(1);
                    }
                    break;
                case 3:
                    do
                    {
                        randomCenter = RandomPointInQuad(points[sourceRightCuff.Item1], points[sourceRightHem.Item1], points[sourceLeftCuff.Item1], points[sourceLeftHem.Item1]);
                    }
                    while (Vector3.Distance(randomCenter, points[sourceLeftHem.Item1]) < 0.2f);
                    yield return StartCoroutine(Fold(points[sourceLeftHem.Item1], randomCenter));
                    points = GetParticles(false);
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        best.Add(1);
                        best.Add(1);
                    }
                    break;
            }
        }
        else
        {
            int edge1 = UnityEngine.Random.Range(0, 4);
            int edge2;
            do
            {
                edge2 = UnityEngine.Random.Range(0, 4);
            }
            while (edge1 == edge2);
            Vector3 grabPoint1;
            Vector3 grabPoint2;
            float rightCuffMaxHeight;
            float leftCuffMaxHeight;
            float rightHemMaxHeight;
            float leftHemMaxHeight;
            float leftShoulderMaxHeight;
            float rightShoulderMaxHeight;
            switch (edge1, edge2)
            {
                case (0, 1) or (1, 0):
                    grabPoint1 = Vector3.Lerp(points[sourceRightShoulder.Item1], points[sourceLeftShoulder.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    grabPoint2 = Vector3.Lerp(points[sourceLeftShoulder.Item1], points[sourceLeftHem.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, true));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                    }
                    break;
                case (1, 2) or (2, 1):
                    grabPoint1 = Vector3.Lerp(points[sourceLeftShoulder.Item1], points[sourceLeftHem.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    grabPoint2 = Vector3.Lerp(points[sourceLeftHem.Item1], points[sourceRightHem.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, true));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                    }

                    break;
                case (2, 3) or (3, 2):
                    grabPoint1 = Vector3.Lerp(points[sourceLeftHem.Item1], points[sourceRightHem.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    grabPoint2 = Vector3.Lerp(points[sourceRightHem.Item1], points[sourceRightShoulder.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, true));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                    }

                    break;
                case (3, 0) or (0, 3):
                    grabPoint1 = Vector3.Lerp(points[sourceRightHem.Item1], points[sourceRightShoulder.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    grabPoint2 = Vector3.Lerp(points[sourceRightShoulder.Item1], points[sourceLeftShoulder.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, true));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);

                        best.Add(1);
                        best.Add(1);
                        best.Add(0);
                        best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                    }
                    break;
                case (2, 0) or (0, 2):
                    grabPoint1 = Vector3.Lerp(points[sourceRightShoulder.Item1], points[sourceLeftShoulder.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    grabPoint2 = Vector3.Lerp(points[sourceRightHem.Item1], points[sourceLeftHem.Item1], UnityEngine.Random.Range(0.1F, 0.9F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, UnityEngine.Random.value > 0.5f));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                    }

                    break;
                case (1, 3) or (3, 1):
                    grabPoint1 = Vector3.Lerp(points[sourceRightShoulder.Item1], points[sourceRightHem.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    grabPoint2 = Vector3.Lerp(points[sourceLeftShoulder.Item1], points[sourceLeftHem.Item1], UnityEngine.Random.Range(0.2F, 0.8F));
                    yield return StartCoroutine(SimpleFling(grabPoint1, grabPoint2, 0.6f, 0.4f, UnityEngine.Random.value > 0.5f));
                    points = GetParticles(false);
                    rightCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightCuff.Item1].x, points[sourceRightCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftCuffMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftCuff.Item1].x, points[sourceLeftCuff.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightHem.Item1].x, points[sourceRightHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftHemMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftHem.Item1].x, points[sourceLeftHem.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    leftShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceLeftShoulder.Item1].x, points[sourceLeftShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    rightShoulderMaxHeight = points.Where((s) => Vector2.Distance(new Vector2(s.x, s.z), new Vector2(points[sourceRightShoulder.Item1].x, points[sourceRightShoulder.Item1].z)) < 0.01f).Select(s => s.y).Max();
                    keyPoint.Clear();
                    best.Clear();
                    if (is_short_sleeve)
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                    }
                    else
                    {
                        keyPoint.Add(points[sourceRightCuff.Item1]);
                        keyPoint.Add(points[sourceLeftCuff.Item1]);
                        keyPoint.Add(points[sourceRightHem.Item1]);
                        keyPoint.Add(points[sourceLeftHem.Item1]);

                        if (rightCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftCuffMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (rightHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);
                        if (leftHemMaxHeight < averagerHeight * 1.5) best.Add(1); else best.Add(0);

                        keyPoint.Add(points[sourceLeftShoulder.Item1]);
                        keyPoint.Add(points[sourceRightShoulder.Item1]);
                        if (leftShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                        if (rightShoulderMaxHeight < averagerHeight * 2.0) best.Add(1); else best.Add(0);
                    }

                    break;
            }
        }
        //Vector3 clothCenter = points[sourceCenter.Item1];
        //Vector3 centerOffset = new Vector3(-clothCenter.x, 0, -clothCenter.z);
        //GetDepthOffset(intrinsicMatrix, centerOffset);
        //List<Vector3> offsetPoint = points.Select(s => s + centerOffset).ToList();
        //List<Vector3> offsetBestPoint = keyPoint.Select(s => s + centerOffset).ToList();
        //RFUniverse.Manager.AssetManager.Instance.SendMessage("Particles", RFUniverseUtility.ListVector3ToListFloat(offsetPoint.ToList()));
        RFUniverse.Manager.AssetManager.Instance.SendMessage("BestGraspPoints", RFUniverseUtility.ListVector3ToListFloat(keyPoint.ToList()), best);
        RFUniverse.Manager.AssetManager.Instance.SendMessage("Done");
        yield break;
    }
    Vector3 RandomPointInQuad(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 v4)
    {
        return UnityEngine.Random.value > 0.5f ? RandomPointInTriangle(v1, v2, v3) : RandomPointInTriangle(v2, v3, v4);
    }
    Vector3 RandomPointInTriangle(Vector3 v1, Vector3 v2, Vector3 v3)
    {
        float a = UnityEngine.Random.Range(0.0f, 1.0f);
        float b = UnityEngine.Random.Range(0.0f, 1.0f);
        if (a + b > 1)
        {
            a = 1 - a;
            b = 1 - b;
        }
        float c = 1 - a - b;
        return v1 * a + v2 * b + v3 * c;
    }
    IEnumerator Fold(Vector3 grabPoint, Vector3 endPoint)
    {
        Vector3 centerPoint = (grabPoint + endPoint) / 2 + Vector3.Distance(grabPoint, endPoint) / 2 * 0.7f * Vector3.up;
        GameObject sphere = AddAttach(grabPoint);
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * lowGravity;
        var tween = sphere.transform.DOMove(centerPoint, 2).SetEase(Ease.InOutSine);
        yield return tween.WaitForCompletion();
        tween = sphere.transform.DOMove(endPoint, 2).SetEase(Ease.InOutSine);
        yield return tween.WaitForCompletion();
        RemoveAllAttach();
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * highGravity;
        yield return new WaitForSeconds(1);
    }

    IEnumerator SimpleFling(Vector3 grabPoint1, Vector3 grabPoint2, float height, float extent, bool dirction)
    {
        GameObject sphere1 = AddAttach(grabPoint1);
        GameObject sphere2 = AddAttach(grabPoint2);
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * lowGravity;
        var tween = sphere1.transform.DOMoveY(height, 2).SetEase(Ease.InOutSine); ;
        sphere2.transform.DOMoveY(height, 2).SetEase(Ease.InOutSine); ;
        yield return tween.WaitForCompletion();
        Vector3 line = sphere2.transform.position - sphere1.transform.position;
        Vector3 dir = Vector3.Cross(Vector3.down, line).normalized * (dirction ? 1 : -1) * extent;
        tween = sphere1.transform.DOMove(dir, 1.5f).SetEase(Ease.InOutSine).SetRelative(true);
        sphere2.transform.DOMove(dir, 1.5f).SetEase(Ease.InOutSine).SetRelative(true);
        yield return tween.WaitForCompletion();
        tween = sphere1.transform.DOMove(grabPoint1, 2f).SetEase(Ease.InOutSine);
        sphere2.transform.DOMove(grabPoint2, 2f).SetEase(Ease.InOutSine);
        yield return tween.WaitForCompletion();
        RemoveAllAttach();
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * highGravity;
        yield return new WaitForSeconds(1);
    }

    float GetMeshPointAverageDistance(Mesh mesh)
    {
        Vector3[] points = mesh.vertices;
        int[] triangles = mesh.triangles;
        List<float> dis = new List<float>();
        for (int i = 0; i < mesh.triangles.Length; i += 3)
        {
            dis.Add(Vector3.Distance(points[triangles[i]], points[triangles[i + 1]]));
            dis.Add(Vector3.Distance(points[triangles[i]], points[triangles[i + 2]]));
            dis.Add(Vector3.Distance(points[triangles[i + 1]], points[triangles[i + 2]]));
        }
        float average = dis.Average();
        particleRadius = average / 2.0f;
        Debug.Log($"GetMeshPointAverageDistance: {average}");
        return average;
    }
    public GameObject AddAttach(Vector3 position)
    {
        Debug.Log("AddAttach");
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        spheres.Add(sphere);
        sphere.transform.position = position;
        sphere.transform.localScale = Vector3.one * 0.03f;
        Destroy(sphere.GetComponent<Collider>());
        currentCloth.AddAttach(sphere.transform);
        return sphere;
    }

    public void RemoveAllAttach()
    {
        Debug.Log("RemoveAllAttach");
        currentCloth.RemoveAllAttach();
        foreach (var item in spheres)
        {
            Destroy(item);
        }
        spheres.Clear();
    }

    public void GetDepth(List<float> intrinsicMatrix)
    {
        Debug.Log("GetDepth");
        depthMesh.gameObject.SetActive(true);
        Mesh mesh = currentCloth.MyGetMesh();
        depthMesh.mesh = mesh;
        for (int i = 0; i < cameras.Length; i++)
        {
            float[,] matrix = new float[3, 3];
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    matrix[j, k] = intrinsicMatrix[j * 3 + k];
                }
            }
            Texture2D tex = cameras[i].GetDepthEXR(matrix);
            RFUniverse.Manager.AssetManager.Instance.SendMessage("Depth", i, Convert.ToBase64String(tex.EncodeToEXR(Texture2D.EXRFlags.CompressRLE)));
            //File.WriteAllBytes("D:\\Desktop\\img.exr", texs[i].EncodeToEXR(Texture2D.EXRFlags.CompressRLE));
        }
        depthMesh.gameObject.SetActive(false);
    }

    public void GetInitParticles(float minZ = 0)
    {
        Debug.Log("GetInitParticles");
        Vector3[] positions = GetParticles(false);
        Quaternion rotation = Quaternion.AngleAxis(-currentCloth.transform.eulerAngles.y, Vector3.up);
        positions = positions.Select((s) => rotation * s).ToArray();
        rotation = Quaternion.AngleAxis(180, Vector3.up);
        positions = positions.Select((s) => rotation * s).ToArray();
        float currentMinZ = positions.Select((s) => s.z).Min();
        float currentMaxX = positions.Select((s) => s.x).Max();
        float currentMinX = positions.Select((s) => s.x).Min();
        float currentCenterX = (currentMaxX + currentMinX) / 2;
        positions = positions.Select((s) => s + new Vector3(-currentCenterX, 0, minZ - currentMinZ)).ToArray();
        initPoints = positions;
        RFUniverse.Manager.AssetManager.Instance.SendMessage("InitParticles", RFUniverseUtility.ListVector3ToListFloat(positions.ToList()));
    }
    public Vector3[] GetParticles(bool send = true)
    {
        Debug.Log("GetParticles");
        Mesh mesh = currentCloth.MyGetMesh();
        Vector3[] positions = mesh.vertices;
        if (send)
            RFUniverse.Manager.AssetManager.Instance.SendMessage("Particles", RFUniverseUtility.ListVector3ToListFloat(positions.ToList()));
        return positions;
    }

    public void GetGraspPosition()
    {
        Debug.Log("GetGraspPosition");
        RFUniverse.Manager.AssetManager.Instance.SendMessage("GraspPoint", sphere1.transform.position.x, sphere1.transform.position.y, sphere1.transform.position.z,
                                                                                                                            sphere2.transform.position.x, sphere2.transform.position.y, sphere2.transform.position.z);
    }

    List<GameObject> spheres = new List<GameObject>();
    public IEnumerator Grab(Vector3 grabPoint, Vector3 endPoint)
    {
        Debug.Log("Grab");

        // TODO: add random rotation, random height[0.5m-1.5m] and random translation[0-0.2m]
        //Vector3[] positions = GetParticles();
        //Vector3 position = positions[UnityEngine.Random.Range(0, positions.Length)];
        //position.y += 0.1f;
        GameObject sphere = AddAttach(grabPoint);
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * lowGravity;
        sphere.transform.DOMove(endPoint, 3).onComplete += () =>
        {
            RemoveAllAttach();
        };
        yield return new WaitForSeconds(5);
        currentClothDynamicsV2._globalSimParams.gravity = Vector3.down * highGravity;
        yield return new WaitForSeconds(1);
        RFUniverse.Manager.AssetManager.Instance.SendMessage("Done");

    }

    GameObject sphere1;
    GameObject sphere2;
    public IEnumerator Stretch(Vector3 position1, Vector3 position2,
        float height, float difference, float z_offset, float max_stretch_distance)
    {
        Debug.Log("Stretch");
        sphere1 = AddAttach(position1);
        sphere2 = AddAttach(position2);
        float dis = (position1 - position2).magnitude;
        sphere1.transform.DOMove(Vector3.up * height + Vector3.left * dis / 2 + Vector3.forward * z_offset, 3);
        sphere2.transform.DOMove(Vector3.up * height + Vector3.right * dis / 2 + Vector3.forward * z_offset, 3);
        yield return new WaitForSeconds(3);
        do
        {
            sphere1.transform.DOMove(Vector3.left * 0.02f, 0.2f).SetRelative(true).SetEase(Ease.Linear);
            sphere2.transform.DOMove(Vector3.right * 0.02f, 0.2f).SetRelative(true).SetEase(Ease.Linear);
            yield return new WaitForSeconds(0.2f);
        }
        while ((GetCenterHeightDifference(Vector3.up * height) > difference * Vector3.Distance(sphere1.transform.position, sphere2.transform.position)
            ) && Vector3.Distance(sphere1.transform.position, sphere2.transform.position) < max_stretch_distance);
        //while (true)
        //{
        //    float stretchLength = GetStretchLength(sphere1.transform.position, sphere2.transform.position, difference);
        //    if (stretchLength == 0) break;
        //    sphere1.transform.DOMove(Vector3.left * stretchLength, 1).SetRelative(true).SetEase(Ease.Linear);
        //    sphere2.transform.DOMove(Vector3.right * stretchLength, 1).SetRelative(true).SetEase(Ease.Linear);
        //    yield return new WaitForSeconds(1);
        //}
        RFUniverse.Manager.AssetManager.Instance.SendMessage("Done");
        yield return null;
    }

    public void DualMove(Vector3 endPoint1, Vector3 endPoint2, float speed)
    {
        Debug.Log("DualMove");
        sphere1.transform.DOMove(endPoint1, speed).SetSpeedBased(true).SetEase(Ease.InOutSine);
        sphere2.transform.DOMove(endPoint2, speed).SetSpeedBased(true).SetEase(Ease.InOutSine).onComplete += () =>
        {
            RFUniverse.Manager.AssetManager.Instance.SendMessage("Done");
        };
    }

    float GetStretchLength(Vector3 position1, Vector3 position2, float difference)
    {
        Vector3 center = (position1 + position2) / 2;
        Vector3[] positions = GetParticles(false);
        float centerMaxY = positions.Where((s) => Mathf.Abs(s.x - center.x) < 0.03f).Select((s) => s.y).Max();
        float a = Mathf.Max(center.y - centerMaxY, 0);
        if (a < difference)
            return 0;
        float b = center.x - position1.x;
        float c = Mathf.Sqrt(a * a + b * b);
        return c - b;
    }

    float GetCenterHeightDifference(Vector3 center)
    {
        Vector3[] positions = GetParticles(false);
        float centerMaxY = positions.Where((s) => Mathf.Abs(s.x - center.x) < 0.02f).Select((s) => s.y).Max();
        return center.y - centerMaxY;
    }

    Vector3[] initPoints = new Vector3[0];
    private void OnDrawGizmos()
    {
        foreach (var point in initPoints)
        {
            Gizmos.color = Color.white;
            Gizmos.DrawSphere(point, 0.005f);
        }
        if (sourceRightCuff != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceRightCuff.Item2, 0.01f);
        }
        if (sourceLeftCuff != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceLeftCuff.Item2, 0.01f);
        }
        if (sourceRightHem != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceRightHem.Item2, 0.01f);
        }
        if (sourceLeftHem != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceLeftHem.Item2, 0.01f);
        }
        if (sourceRightShoulder != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceRightShoulder.Item2, 0.01f);
        }
        if (sourceLeftShoulder != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceLeftShoulder.Item2, 0.01f);
        }
        if (sourceCenter != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(sourceCenter.Item2, 0.01f);
        }
        for (int i = 0; i < keyPoint.Count; i++)
        {
            if (best[i] == 1)
            {
                Gizmos.color = Color.blue;
                Gizmos.DrawSphere(keyPoint[i], 0.01f);
            }
        }
    }

}

#if UNITY_EDITOR
[CustomEditor(typeof(Main), true)]
public class MainEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        Main script = target as Main;
        GUILayout.Space(10);
        GUILayout.Label("Editor Tool:");
        if (GUILayout.Button("LoadMesh"))
        {
            script.StartCoroutine(script.LoadMesh("./val_t1/07619/Tshirt.obj", 100, UnityEngine.Random.Range(-180f, 180f)));
        }
        if (GUILayout.Button("Grab"))
        {
            script.StartCoroutine(script.Grab(script.GetRandomPosition(), Vector3.up * 0.5f));
        }
        if (GUILayout.Button("RandomFold"))
        {
            script.StartCoroutine(script.RandomFold(false));  // long sleeve
        }
        if (GUILayout.Button("Stretch"))
        {
            script.StartCoroutine(script.Stretch(script.GetRandomPosition(), script.GetRandomPosition(), 1, 0.1f, 0f, 1.25f));
        }
        if (GUILayout.Button("GetDepth"))
        {
            script.GetDepth(new List<float>() { 1728.28f, 0, 0, 0, 1728.25f, 0, 821.533f, 593.029f, 1f });
        }
    }
}
#endif
